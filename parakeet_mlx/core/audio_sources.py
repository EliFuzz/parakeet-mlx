import asyncio
import logging
import queue
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, Generator, Optional, Union

import mlx.core as mx
import numpy as np
import sounddevice as sd

from parakeet_mlx.audio.noise_reduction import create_noise_reducer
from parakeet_mlx.audio.processing import PreprocessArgs, load_audio
from parakeet_mlx.core.config import AudioConfig, AudioSource

logger = logging.getLogger(__name__)


class SilenceDetector:
    """Detects silence in audio streams for dynamic speech boundary detection"""

    def __init__(self,
                 sample_rate: int,
                 silence_threshold: float = 0.01,
                 silence_duration_ms: int = 200,
                 min_speech_duration_ms: int = 100):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration_samples = int(silence_duration_ms * sample_rate / 1000)
        self.min_speech_duration_samples = int(min_speech_duration_ms * sample_rate / 1000)

        # State tracking
        self.consecutive_silence_samples = 0
        self.total_speech_samples = 0
        self.is_speech_detected = False

    def add_audio(self, audio_chunk: mx.array) -> bool:
        """
        Add audio chunk and check if silence boundary is detected.

        Args:
            audio_chunk: Audio data as MLX array

        Returns:
            bool: True if silence boundary detected (should process buffer)
        """
        # Calculate RMS energy of the chunk
        audio_np = np.array(audio_chunk)
        rms_energy = np.sqrt(np.mean(audio_np ** 2))

        if rms_energy > self.silence_threshold:
            # Speech detected
            self.consecutive_silence_samples = 0
            self.total_speech_samples += len(audio_chunk)
            self.is_speech_detected = True
        else:
            # Silence detected
            self.consecutive_silence_samples += len(audio_chunk)

        # Check if we should trigger transcription
        if (self.is_speech_detected and
            self.consecutive_silence_samples >= self.silence_duration_samples and
            self.total_speech_samples >= self.min_speech_duration_samples):
            return True

        return False

    def reset(self):
        """Reset detector state after processing"""
        self.consecutive_silence_samples = 0
        self.total_speech_samples = 0
        self.is_speech_detected = False


class AudioSourceBase(ABC):
    """Abstract base class for audio input sources"""
    
    def __init__(self, config: AudioConfig, preprocess_args: PreprocessArgs):
        self.config = config
        self.preprocess_args = preprocess_args
        self._is_active = False
    
    @abstractmethod
    def get_audio_data(self) -> mx.array:
        """Get audio data as MLX array"""
        pass
    
    @abstractmethod
    def get_audio_stream(self) -> Generator[mx.array, None, None]:
        """Get audio data as a stream of chunks"""
        pass
    
    async def get_audio_stream_async(self) -> AsyncGenerator[mx.array, None]:
        """Get audio data as an async stream of chunks"""
        for chunk in self.get_audio_stream():
            yield chunk
            await asyncio.sleep(0)  # Allow other coroutines to run
    
    def start(self):
        """Start the audio source"""
        self._is_active = True
    
    def stop(self):
        """Stop the audio source"""
        self._is_active = False
    
    @property
    def is_active(self) -> bool:
        """Check if the audio source is active"""
        return self._is_active


class FileAudioSource(AudioSourceBase):
    """Audio source for file-based input"""
    
    def __init__(self, config: AudioConfig, preprocess_args: PreprocessArgs):
        super().__init__(config, preprocess_args)
        if config.file_path is None:
            raise ValueError("file_path is required for FileAudioSource")
        self.file_path = Path(config.file_path)
        self._audio_data = None
    
    def get_audio_data(self) -> mx.array:
        """Load and return the entire audio file"""
        if self._audio_data is None:
            dtype = mx.bfloat16 if self.config.dtype == "bfloat16" else mx.float32
            self._audio_data = load_audio(
                self.file_path, 
                self.preprocess_args.sample_rate, 
                dtype
            )
        return self._audio_data
    
    def get_audio_stream(self) -> Generator[mx.array, None, None]:
        """Stream audio data in chunks"""
        audio_data = self.get_audio_data()
        
        if self.config.chunk_duration is None:
            # Return entire file as single chunk
            yield audio_data
            return
        
        chunk_samples = int(self.config.chunk_duration * self.preprocess_args.sample_rate)
        overlap_samples = int(self.config.overlap_duration * self.preprocess_args.sample_rate)
        
        start = 0
        while start < len(audio_data):
            end = min(start + chunk_samples, len(audio_data))
            chunk = audio_data[start:end]
            yield chunk
            
            if end >= len(audio_data):
                break
            
            start = end - overlap_samples


class MicrophoneAudioSource(AudioSourceBase):
    """Audio source for real-time microphone input"""

    def __init__(self, config: AudioConfig, preprocess_args: PreprocessArgs):
        super().__init__(config, preprocess_args)
        self._audio_queue = queue.Queue()
        self._stream = None
        self._recording_thread = None

        # Validate microphone device
        self._device_id = self._get_device_id()

        # Noise reduction setup
        self._noise_reducer = create_noise_reducer(
            sample_rate=preprocess_args.sample_rate,
            noise_reduction_strength=config.noise_reduction,
            enable_noise_reduction=config.noise_reduction > 0.0
        )

        # Silence detection setup
        if config.enable_silence_detection:
            self._silence_detector = SilenceDetector(
                sample_rate=preprocess_args.sample_rate,
                silence_threshold=config.silence_threshold,
                silence_duration_ms=config.silence_duration_ms,
                min_speech_duration_ms=config.min_speech_duration_ms
            )
            self._audio_buffer = []
            self._buffer_start_time = None
            self._max_buffer_duration = config.max_buffer_duration_s
        else:
            self._silence_detector = None
            self._audio_buffer = None
    
    def _get_device_id(self) -> Optional[int]:
        """Get the microphone device ID"""
        if self.config.microphone_device is None:
            return None  # Use default device
        
        if isinstance(self.config.microphone_device, int):
            return self.config.microphone_device
        
        # Search by device name
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if self.config.microphone_device.lower() in device['name'].lower():
                if device['max_input_channels'] > 0:
                    return i
        
        raise ValueError(f"Microphone device not found: {self.config.microphone_device}")
    
    def _audio_callback(self, indata, frames, callback_time, status):
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio callback status: {status}")

        # Convert to MLX array
        audio_chunk = mx.array(indata[:, 0].copy())  # Take first channel

        # Apply noise reduction before silence detection
        audio_chunk = self._noise_reducer.process_audio_chunk(audio_chunk)

        if self._silence_detector is not None:
            # Handle silence detection
            if self._buffer_start_time is None:
                self._buffer_start_time = time.time()

            self._audio_buffer.append(audio_chunk)

            # Check for silence boundary or max buffer duration
            silence_detected = self._silence_detector.add_audio(audio_chunk)
            buffer_duration = time.time() - self._buffer_start_time

            if silence_detected or buffer_duration >= self._max_buffer_duration:
                # Concatenate buffered audio and put in queue
                if self._audio_buffer:
                    combined_audio = mx.concatenate(self._audio_buffer, axis=0)
                    self._audio_queue.put(combined_audio)

                    # Reset buffer and detector
                    self._audio_buffer = []
                    self._silence_detector.reset()
                    self._buffer_start_time = None
        else:
            # Original behavior - immediate processing
            self._audio_queue.put(audio_chunk)
    
    def start(self):
        """Start microphone recording"""
        super().start()
        
        try:
            self._stream = sd.InputStream(
                device=self._device_id,
                channels=self.config.channels,
                samplerate=self.preprocess_args.sample_rate,
                blocksize=self.config.buffer_size,
                callback=self._audio_callback,
                dtype=np.float32
            )
            self._stream.start()
            logger.info(f"Started microphone recording on device {self._device_id}")
        except Exception as e:
            logger.error(f"Failed to start microphone: {e}")
            raise
    
    def stop(self):
        """Stop microphone recording"""
        super().stop()
        
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Stopped microphone recording")
    
    def get_audio_data(self) -> mx.array:
        """Not supported for real-time microphone input"""
        raise NotImplementedError("get_audio_data not supported for microphone input")
    
    def get_audio_stream(self) -> Generator[mx.array, None, None]:
        """Stream audio data from microphone"""
        if not self._is_active:
            self.start()

        try:
            while self._is_active:
                try:
                    chunk = self._audio_queue.get(timeout=1.0)
                    yield chunk
                except queue.Empty:
                    continue
        finally:
            self.stop()

    def clear_audio_buffer(self):
        """Clear any remaining audio in the internal buffer (for silence detection)"""
        if self._silence_detector is not None and self._audio_buffer:
            self._audio_buffer = []
            self._silence_detector.reset()
            self._buffer_start_time = None


class StreamAudioSource(AudioSourceBase):
    """Audio source for streaming data input"""
    
    def __init__(self, config: AudioConfig, preprocess_args: PreprocessArgs):
        super().__init__(config, preprocess_args)
        self._stream_queue = queue.Queue()
    
    def add_audio_chunk(self, audio_chunk: Union[mx.array, np.ndarray]):
        """Add an audio chunk to the stream"""
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = mx.array(audio_chunk)
        self._stream_queue.put(audio_chunk)
    
    def get_audio_data(self) -> mx.array:
        """Not supported for streaming input"""
        raise NotImplementedError("get_audio_data not supported for streaming input")
    
    def get_audio_stream(self) -> Generator[mx.array, None, None]:
        """Stream audio data from queue"""
        while self._is_active:
            try:
                chunk = self._stream_queue.get(timeout=1.0)
                yield chunk
            except queue.Empty:
                continue


def create_audio_source(
    config: AudioConfig, 
    preprocess_args: PreprocessArgs
) -> AudioSourceBase:
    """Factory function to create appropriate audio source"""
    
    if config.source == AudioSource.FILE:
        return FileAudioSource(config, preprocess_args)
    elif config.source == AudioSource.MICROPHONE:
        return MicrophoneAudioSource(config, preprocess_args)
    elif config.source == AudioSource.STREAM:
        return StreamAudioSource(config, preprocess_args)
    else:
        raise ValueError(f"Unsupported audio source: {config.source}")


# Utility functions for audio device management
def list_audio_devices() -> list:
    """List available audio input devices"""
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append({
                'id': i,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'sample_rate': device['default_samplerate']
            })
    
    return input_devices


def get_default_input_device() -> dict:
    """Get the default audio input device"""
    default_device = sd.query_devices(kind='input')
    return {
        'id': sd.default.device[0] if sd.default.device[0] is not None else 0,
        'name': default_device['name'],
        'channels': default_device['max_input_channels'],
        'sample_rate': default_device['default_samplerate']
    }
