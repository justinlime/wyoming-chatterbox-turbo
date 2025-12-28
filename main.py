#!/usr/bin/env python3
"""
Wyoming Protocol TTS Server for Chatterbox-Turbo
Home Assistant compatible real-time text-to-speech service with audio post-processing
"""
import argparse
import asyncio
import logging
import os
import queue
import re
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnxruntime
import librosa
import soundfile as sf
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize
from wyoming.info import Describe

# Optional scipy for audio filters
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Model constants
MODEL_ID = "ResembleAI/chatterbox-turbo-ONNX"
SAMPLE_RATE = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SILENCE_TOKEN = 4299
NUM_KV_HEADS = 16
HEAD_DIM = 64

_LOGGER = logging.getLogger(__name__)


class AudioPostProcessor:
    """
    Lightweight audio post-processing for TTS output.
    
    Applies minimal overhead processing (~5-10ms per chunk):
    - High-pass filter (removes DC offset and low-frequency rumble)
    - Soft clipping (prevents harsh distortion)
    - Peak normalization (consistent volume levels)
    """
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        highpass_cutoff: float = 80.0,
        target_peak: float = 0.95,
    ):
        """
        Initialize audio post-processor with sane defaults.
        
        Args:
            sample_rate: Sample rate of audio (default: 24000 Hz)
            highpass_cutoff: High-pass filter cutoff in Hz (default: 80 Hz)
            target_peak: Target peak normalization level (default: 0.95)
        """
        self.sample_rate = sample_rate
        self.target_peak = target_peak
        self.enabled = True
        
        # Pre-compute high-pass filter coefficients if scipy available
        if SCIPY_AVAILABLE:
            nyquist = sample_rate / 2
            normalized_cutoff = highpass_cutoff / nyquist
            self.hp_b, self.hp_a = signal.butter(2, normalized_cutoff, btype='high')
            _LOGGER.info(f"Audio post-processing enabled: {highpass_cutoff}Hz HPF, {target_peak} peak")
        else:
            self.hp_b = None
            self.hp_a = None
            _LOGGER.warning("SciPy not available - high-pass filter disabled")
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to audio chunk.
        
        Args:
            audio: Input audio (float32, -1.0 to 1.0)
            
        Returns:
            Processed audio (float32)
        """
        if not self.enabled or audio is None or len(audio) == 0:
            return audio
        
        # Ensure float32 for consistent processing
        processed = audio.astype(np.float32, copy=True)
        
        # 1. High-pass filter (remove DC offset and rumble)
        if SCIPY_AVAILABLE and self.hp_b is not None:
            try:
                # Use filtfilt for zero-phase filtering (no delay)
                processed = signal.filtfilt(self.hp_b, self.hp_a, processed).astype(np.float32)
            except Exception as e:
                _LOGGER.debug(f"High-pass filter failed: {e}")
        
        # 2. Soft clipping (prevents harsh distortion)
        # Tanh-based soft limiter with gentle knee
        processed = np.tanh(processed * 1.2) * 0.9
        
        # 3. Peak normalization (consistent volume)
        peak = np.abs(processed).max()
        if peak > 1e-6:  # Avoid division by zero
            processed = processed * (self.target_peak / peak)
        
        return processed


class RepetitionPenaltyLogitsProcessor:
    """Apply repetition penalty to prevent model from repeating tokens"""
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"penalty must be positive float, got {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        np.put_along_axis(scores, input_ids, score, axis=1)
        return scores


class ChatterboxTTS:
    """Chatterbox-Turbo TTS inference engine"""
    
    def __init__(self, dtype: str = "fp32", voices_dir: str = "./voices", use_cuda: bool = False):
        self.dtype = dtype
        self.voices_dir = Path(voices_dir)
        self.voices: Dict[str, np.ndarray] = {}
        self.voice_cache: Dict[str, tuple] = {}  # Cache encoded voices
        self.tokenizer = None
        self.use_cuda = use_cuda
        self.max_text_length = 200  # Keep chunks small for consistent VRAM usage
        
        # Audio post-processor (initialized after we know sample rate)
        self.post_processor = AudioPostProcessor(sample_rate=SAMPLE_RATE)
        
        # Configure execution providers for ONNX
        if self.use_cuda:
            self.providers = [
                ('CUDAExecutionProvider', {
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cudnn_conv_algo_search': 'DEFAULT',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ]
            _LOGGER.info("CUDA acceleration enabled")
        else:
            self.providers = ['CPUExecutionProvider']
            _LOGGER.info("Using CPU execution")
        
        # ONNX sessions
        self.speech_encoder_session = None
        self.embed_tokens_session = None
        self.language_model_session = None
        self.cond_decoder_session = None
        
    def download_model(self, name: str) -> str:
        """Download ONNX model from HuggingFace"""
        suffix = "" if self.dtype == "fp32" else "_quantized" if self.dtype == "q8" else f"_{self.dtype}"
        filename = f"{name}{suffix}.onnx"
        _LOGGER.info(f"Downloading {filename}...")
        graph = hf_hub_download(MODEL_ID, subfolder="onnx", filename=filename)
        hf_hub_download(MODEL_ID, subfolder="onnx", filename=f"{filename}_data")
        return graph
    
    def load_models(self):
        """Load all ONNX models and tokenizer"""
        _LOGGER.info("Loading models...")
        
        # Download models
        conditional_decoder_path = self.download_model("conditional_decoder")
        speech_encoder_path = self.download_model("speech_encoder")
        embed_tokens_path = self.download_model("embed_tokens")
        language_model_path = self.download_model("language_model")
        
        # Create ONNX sessions with optimizations
        _LOGGER.info("Creating ONNX inference sessions...")
        
        # Suppress ONNX Runtime warnings
        onnxruntime.set_default_logger_severity(3)  # 3 = Error, suppresses warnings
        
        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3  # 3 = Error
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        
        if self.use_cuda:
            sess_options.enable_mem_pattern = False  # Reduce memory fragmentation
            sess_options.enable_cpu_mem_arena = False
            sess_options.enable_mem_reuse = True  # Aggressive memory reuse
        
        self.speech_encoder_session = onnxruntime.InferenceSession(
            speech_encoder_path, 
            sess_options=sess_options,
            providers=self.providers
        )
        self.embed_tokens_session = onnxruntime.InferenceSession(
            embed_tokens_path,
            sess_options=sess_options,
            providers=self.providers
        )
        self.language_model_session = onnxruntime.InferenceSession(
            language_model_path,
            sess_options=sess_options,
            providers=self.providers
        )
        self.cond_decoder_session = onnxruntime.InferenceSession(
            conditional_decoder_path,
            sess_options=sess_options,
            providers=self.providers
        )
        
        # Warm up models
        if self.use_cuda:
            _LOGGER.info("Warming up GPU models...")
            dummy_audio = np.zeros((1, SAMPLE_RATE), dtype=np.float32)
            dummy_ids = np.array([[1, 2, 3]], dtype=np.int64)
            try:
                self.speech_encoder_session.run(None, {"audio_values": dummy_audio})
                self.embed_tokens_session.run(None, {"input_ids": dummy_ids})
            except Exception as e:
                _LOGGER.debug(f"Warmup failed (expected): {e}")
        
        # Log which provider is actually being used
        for name, session in [
            ("speech_encoder", self.speech_encoder_session),
            ("embed_tokens", self.embed_tokens_session),
            ("language_model", self.language_model_session),
            ("cond_decoder", self.cond_decoder_session)
        ]:
            provider = session.get_providers()[0]
            _LOGGER.debug(f"{name} using: {provider}")
        
        # Load tokenizer
        _LOGGER.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        _LOGGER.info("Models loaded successfully")
    
    def load_voices(self):
        """Load all voice reference audio files from voices directory"""
        if not self.voices_dir.exists():
            _LOGGER.warning(f"Voices directory not found: {self.voices_dir}")
            self.voices_dir.mkdir(parents=True, exist_ok=True)
            return
        
        _LOGGER.info(f"Loading voices from {self.voices_dir}...")
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        
        for audio_file in self.voices_dir.iterdir():
            if audio_file.suffix.lower() in audio_extensions:
                try:
                    # Load audio and resample to model sample rate
                    audio_values, _ = librosa.load(str(audio_file), sr=SAMPLE_RATE)
                    audio_values = audio_values[np.newaxis, :].astype(np.float32)
                    
                    # Use filename without extension as voice name
                    voice_name = audio_file.stem
                    self.voices[voice_name] = audio_values
                    
                    # Pre-encode voice for faster synthesis
                    _LOGGER.info(f"Pre-encoding voice: {voice_name}")
                    encoded = self.speech_encoder_session.run(None, {"audio_values": audio_values})
                    self.voice_cache[voice_name] = encoded
                    
                    _LOGGER.info(f"Loaded voice: {voice_name} ({audio_file.name})")
                except Exception as e:
                    _LOGGER.error(f"Failed to load {audio_file.name}: {e}")
        
        if not self.voices:
            _LOGGER.warning("No voices loaded. Add .wav files to voices directory.")
    
    def split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences for memory-efficient processing"""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        # Merge very short sentences to avoid too many chunks
        merged = []
        current = ""
        
        for sentence in sentences:
            if len(current) + len(sentence) <= self.max_text_length:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    merged.append(current)
                current = sentence
        
        if current:
            merged.append(current)
        
        return merged if merged else [text]
    
    def synthesize(
        self,
        text: str,
        voice_name: str,
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.2,
        stream_callback = None
    ):
        """
        Synthesize speech from text using specified voice
        
        Args:
            text: Text to synthesize
            voice_name: Name of voice to use (must exist in self.voices)
            max_new_tokens: Maximum tokens to generate
            repetition_penalty: Penalty for repeating tokens
            stream_callback: Optional callback(audio_chunk) for streaming output
            
        Returns:
            Audio waveform as numpy array (only if stream_callback is None)
        """
        if voice_name not in self.voices:
            raise ValueError(f"Voice '{voice_name}' not found. Available: {list(self.voices.keys())}")
        
        # Split long text into sentences to reduce VRAM usage
        sentences = self.split_into_sentences(text)
        
        if len(sentences) > 1:
            _LOGGER.info(f"Split text into {len(sentences)} chunks for memory efficiency")
        
        if stream_callback:
            # Streaming mode - yield each chunk as it's generated
            for i, sentence in enumerate(sentences):
                _LOGGER.debug(f"Processing chunk {i+1}/{len(sentences)}: {sentence[:50]}...")
                chunk_audio = self._synthesize_chunk(sentence, voice_name, max_new_tokens, repetition_penalty)
                
                # Apply post-processing
                chunk_audio = self.post_processor.process(chunk_audio)
                
                stream_callback(chunk_audio)
        else:
            # Buffered mode - concatenate all chunks
            audio_chunks = []
            for i, sentence in enumerate(sentences):
                _LOGGER.debug(f"Processing chunk {i+1}/{len(sentences)}: {sentence[:50]}...")
                chunk_audio = self._synthesize_chunk(sentence, voice_name, max_new_tokens, repetition_penalty)
                
                # Apply post-processing
                chunk_audio = self.post_processor.process(chunk_audio)
                
                audio_chunks.append(chunk_audio)
            
            wav = np.concatenate(audio_chunks)
            _LOGGER.info(f"Synthesized {len(wav) / SAMPLE_RATE:.2f}s total audio from {len(sentences)} chunks")
            return wav
    
    def _synthesize_chunk(
        self,
        text: str,
        voice_name: str,
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.2
    ) -> np.ndarray:
        """Internal method to synthesize a single text chunk"""
        start_time = time.time()
        _LOGGER.debug(f"Synthesizing chunk with voice '{voice_name}': {text[:50]}...")
        
        # Get cached voice encoding
        cond_emb, prompt_token, speaker_embeddings, speaker_features = self.voice_cache[voice_name]
        
        # Tokenize text
        input_ids = self.tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
        
        # Generation loop
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
        
        # Embed text tokens once
        inputs_embeds = self.embed_tokens_session.run(None, {"input_ids": input_ids})[0]
        inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)
        
        # Initialize cache and LLM inputs
        batch_size, seq_len, _ = inputs_embeds.shape
        past_key_values = {
            inp.name: np.zeros(
                [batch_size, NUM_KV_HEADS, 0, HEAD_DIM],
                dtype=np.float16 if inp.type == 'tensor(float16)' else np.float32
            )
            for inp in self.language_model_session.get_inputs()
            if "past_key_values" in inp.name
        }
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        
        token_gen_start = time.time()
        tokens_generated = 0
        
        for i in range(max_new_tokens):
            # Run language model
            outputs = self.language_model_session.run(None, dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **past_key_values,
            ))
            
            logits = outputs[0]
            present_key_values = outputs[1:]
            
            # Apply repetition penalty and sample next token
            logits = logits[:, -1, :]
            next_token_logits = repetition_penalty_processor(generate_tokens, logits)
            next_token = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
            generate_tokens = np.concatenate((generate_tokens, next_token), axis=-1)
            tokens_generated += 1
            
            # Check for stop token
            if next_token[0, 0] == STOP_SPEECH_TOKEN:
                break
            
            # Update for next iteration - only embed the new token
            inputs_embeds = self.embed_tokens_session.run(None, {"input_ids": next_token})[0]
            attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
            position_ids = position_ids[:, -1:] + 1
            
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]
        
        token_gen_time = time.time() - token_gen_start
        
        # Decode audio from tokens
        decode_start = time.time()
        speech_tokens = generate_tokens[:, 1:-1]
        silence_tokens = np.full((speech_tokens.shape[0], 3), SILENCE_TOKEN, dtype=np.int64)
        speech_tokens = np.concatenate([prompt_token, speech_tokens, silence_tokens], axis=1)
        
        wav = self.cond_decoder_session.run(None, dict(
            speech_tokens=speech_tokens,
            speaker_embeddings=speaker_embeddings,
            speaker_features=speaker_features,
        ))[0].squeeze(axis=0)
        
        decode_time = time.time() - decode_start
        total_time = time.time() - start_time
        audio_duration = len(wav) / SAMPLE_RATE
        rtf = audio_duration / total_time if total_time > 0 else 0
        
        _LOGGER.info(
            f"Generated {audio_duration:.2f}s audio in {total_time:.2f}s "
            f"(RTF: {rtf:.2f}x, {tokens_generated} tokens in {token_gen_time:.2f}s, "
            f"decode: {decode_time:.2f}s)"
        )
        
        return wav


class ChatterboxEventHandler(AsyncEventHandler):
    """Handle Wyoming protocol events for TTS"""
    
    def __init__(
        self,
        tts_engine: ChatterboxTTS,
        wyoming_info: Info,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tts_engine = tts_engine
        self.wyoming_info_event = wyoming_info.event()
        self.default_voice = next(iter(tts_engine.voices.keys())) if tts_engine.voices else None
    
    async def handle_event(self, event: Event) -> bool:
        """Process incoming Wyoming events"""
        if Describe.is_type(event.type):
            # Send server info in response to describe request
            await self.write_event(self.wyoming_info_event)
            return True
        
        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            _LOGGER.debug(f"Received synthesis request: {synthesize.text}")
            
            # Determine which voice to use
            voice_name = synthesize.voice.name if synthesize.voice else self.default_voice
            if not voice_name:
                _LOGGER.error("No voice specified and no default voice available")
                return False
            
            try:
                synthesis_start = time.time()
                first_chunk_sent = False
                total_audio_duration = 0
                
                # Send audio start event immediately
                await self.write_event(
                    AudioStart(
                        rate=SAMPLE_RATE,
                        width=2,
                        channels=1
                    ).event()
                )
                
                # Thread-safe queue for passing audio between threads
                audio_queue = queue.Queue()
                
                # Audio sender task
                async def send_audio():
                    nonlocal first_chunk_sent, total_audio_duration
                    
                    while True:
                        # Non-blocking check for audio chunks
                        try:
                            wav_chunk = await asyncio.to_thread(audio_queue.get, timeout=0.1)
                        except queue.Empty:
                            continue
                        
                        if wav_chunk is None:  # Sentinel for end
                            break
                        
                        # Convert to 16-bit PCM (already post-processed by the engine)
                        wav_int16 = (wav_chunk * 32767).astype(np.int16)
                        total_audio_duration += len(wav_int16) / SAMPLE_RATE
                        
                        # Stream in small chunks
                        chunk_size = 1024
                        for i in range(0, len(wav_int16), chunk_size):
                            chunk = wav_int16[i:i + chunk_size]
                            await self.write_event(
                                AudioChunk(
                                    rate=SAMPLE_RATE,
                                    width=2,
                                    channels=1,
                                    audio=chunk.tobytes()
                                ).event()
                            )
                            
                            if not first_chunk_sent:
                                ttfb = time.time() - synthesis_start
                                _LOGGER.info(f"First audio chunk sent in {ttfb:.3f}s (TTFB)")
                                first_chunk_sent = True
                
                # Start sender task
                sender_task = asyncio.create_task(send_audio())
                
                # Callback to queue audio chunks (thread-safe)
                def queue_callback(wav_chunk):
                    audio_queue.put(wav_chunk)
                
                # Synthesize in background (post-processing applied inside synthesize())
                await asyncio.to_thread(
                    self.tts_engine.synthesize,
                    synthesize.text,
                    voice_name,
                    stream_callback=queue_callback
                )
                
                # Signal end and wait for sender to finish
                audio_queue.put(None)
                await sender_task
                
                # Send audio stop event
                await self.write_event(AudioStop().event())
                
                total_time = time.time() - synthesis_start
                _LOGGER.info(
                    f"Complete synthesis: {total_audio_duration:.2f}s audio, "
                    f"{total_time:.2f}s total, RTF: {total_audio_duration/total_time:.2f}x"
                )
                _LOGGER.debug("Audio streaming complete")
                
            except Exception as e:
                _LOGGER.error(f"Synthesis failed: {e}", exc_info=True)
                return False
        
        return True


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Wyoming Chatterbox TTS Server")
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0, env: HOST)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "10200")),
        help="Port to bind to (default: 10200, env: PORT)"
    )
    parser.add_argument(
        "--voices-dir",
        default=os.environ.get("VOICES_DIR", "./voices"),
        help="Directory containing voice reference audio files (default: ./voices, env: VOICES_DIR)"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=os.environ.get("CUDA", "").lower() in ("1", "true", "yes"),
        help="Use CUDA GPU acceleration (default: False, env: CUDA)"
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["fp32", "q8", "q4"],
        help="Model data type - fp32 for GPU, q8/q4 for CPU (default: auto-select based on CUDA, env: DTYPE)"
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO, env: LOG_LEVEL)"
    )
    
    args = parser.parse_args()
    
    # Auto-select dtype if not specified
    if args.dtype is None:
        args.dtype = os.environ.get("DTYPE")
        if args.dtype is None:
            args.dtype = "fp32" if args.cuda else "q4"
            _LOGGER.info(f"Auto-selected dtype: {args.dtype}")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    _LOGGER.info("=" * 60)
    _LOGGER.info("Wyoming Chatterbox TTS Server")
    _LOGGER.info("=" * 60)
    
    # Check CUDA availability if requested
    if args.cuda:
        cuda_available = 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
        if not cuda_available:
            _LOGGER.warning("CUDA requested but not available. Install onnxruntime-gpu for CUDA support.")
            _LOGGER.warning("Falling back to CPU execution.")
        else:
            _LOGGER.info("CUDA is available and will be used for acceleration")
    
    # Initialize TTS engine
    tts_engine = ChatterboxTTS(
        dtype=args.dtype,
        voices_dir=args.voices_dir,
        use_cuda=args.cuda
    )
    
    try:
        tts_engine.load_models()
        tts_engine.load_voices()
    except Exception as e:
        _LOGGER.error(f"Failed to initialize TTS engine: {e}", exc_info=True)
        return 1
    
    if not tts_engine.voices:
        _LOGGER.error("No voices available. Cannot start server.")
        return 1
    
    # Build Wyoming info
    voices = [
        TtsVoice(
            name=name,
            description=f"Chatterbox voice: {name}",
            attribution=Attribution(
                name="Resemble AI",
                url="https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX"
            ),
            installed=True,
            version="1.0.0",
            languages=["en"]  # Chatterbox supports English
        )
        for name in tts_engine.voices.keys()
    ]
    
    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="chatterbox-turbo",
                description="Resemble AI Chatterbox Turbo TTS",
                attribution=Attribution(
                    name="Resemble AI",
                    url="https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX"
                ),
                installed=True,
                version="1.0.0",
                voices=voices
            )
        ]
    )
    
    # Start Wyoming server
    _LOGGER.info(f"Starting server on {args.host}:{args.port}")
    _LOGGER.info(f"Available voices: {', '.join(tts_engine.voices.keys())}")
    
    server = AsyncServer.from_uri(f"tcp://{args.host}:{args.port}")
    
    # Use partial to bind custom parameters, leaving reader/writer for Wyoming
    await server.run(partial(
        ChatterboxEventHandler,
        tts_engine,
        wyoming_info
    ))
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        _LOGGER.info("Shutting down...")
        sys.exit(0)
