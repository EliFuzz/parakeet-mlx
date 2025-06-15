import asyncio
import logging
import time
from pathlib import Path
from typing import Generator, Optional, Union

import mlx.core as mx

from parakeet_mlx.audio.alignment import AlignedResult
from parakeet_mlx.audio.filler_filtering import FillerFilterConfig, FillerWordFilter
from parakeet_mlx.audio.processing import get_logmel
from parakeet_mlx.core.audio_sources import create_audio_source
from parakeet_mlx.core.config import (
    AudioSource,
    ModelConfig,
    ProcessingMode,
    TranscriptionConfig,
)
from parakeet_mlx.core.performance import PerformanceManager
from parakeet_mlx.utils.model_loading import from_pretrained

logger = logging.getLogger(__name__)


class UnifiedTranscriber:
    
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self.config.setup_logging()
        
        self._model = None
        self._audio_source = None
        self._performance_manager = None
        self._async_processor = None
        self._filler_filter = None
        
        self._is_initialized = False
        self._is_streaming = False
        
        logger.info("Initialized UnifiedTranscriber")
    
    def _ensure_initialized(self):
        if self._is_initialized:
            return
        
        logger.info(f"Loading model: {self.config.model.model_name}")
        dtype = mx.bfloat16 if self.config.model.dtype == "bfloat16" else mx.float32
        self._model = from_pretrained(self.config.model.model_name, dtype=dtype)
        
        self._performance_manager = PerformanceManager(self.config.performance)
        
        self._audio_source = create_audio_source(
            self.config.audio,
            self._model.preprocessor_config
        )

        if self.config.audio.filler_confidence_threshold > 0.0:
            filter_config = FillerFilterConfig(
                enabled=True,
                confidence_threshold=self.config.audio.filler_confidence_threshold,
                custom_filler_words=set(self.config.audio.custom_filler_words) if self.config.audio.custom_filler_words else None
            )
            self._filler_filter = FillerWordFilter(filter_config)

        self._is_initialized = True
        logger.info("Transcriber initialization complete")

    def _apply_filler_filtering(self, result: AlignedResult) -> AlignedResult:
        if self._filler_filter is not None:
            return self._filler_filter.filter_result(result)
        return result

    def transcribe(
        self, 
        audio_source: Optional[Union[str, Path, mx.array]] = None,
        config: Optional[TranscriptionConfig] = None
    ) -> AlignedResult:

        # Use provided config or instance config
        if config is not None:
            original_config = self.config
            self.config = config
            self._is_initialized = False  # Force re-initialization
        
        try:
            self._ensure_initialized()
            
            if audio_source is not None:
                return self._transcribe_input(audio_source)
            else:
                return self._transcribe_from_source()
        
        finally:
            if config is not None:
                self.config = original_config
                self._is_initialized = False
    
    def _transcribe_input(self, audio_source: Union[str, Path, mx.array]) -> AlignedResult:
        if isinstance(audio_source, (str, Path)):
            return self._transcribe_file(Path(audio_source))
        elif isinstance(audio_source, mx.array):
            return self._transcribe_array(audio_source)
        else:
            raise ValueError(f"Unsupported audio source type: {type(audio_source)}")
    
    def _transcribe_from_source(self) -> AlignedResult:
        if self.config.audio.source == AudioSource.FILE:
            if self.config.audio.file_path is None:
                raise ValueError("file_path must be specified for FILE source")
            return self._transcribe_file(self.config.audio.file_path)
        
        elif self.config.audio.source == AudioSource.MICROPHONE:
            return self._transcribe_microphone()
        
        elif self.config.audio.source == AudioSource.STREAM:
            raise ValueError("Use transcribe_stream() for streaming transcription")
        
        else:
            raise ValueError(f"Unsupported audio source: {self.config.audio.source}")
    
    def _transcribe_file(self, file_path: Path) -> AlignedResult:
        logger.info(f"Transcribing file: {file_path}")
        
        if self.config.mode == ProcessingMode.BATCH:
            return self._model.transcribe(
                file_path,
                dtype=mx.bfloat16 if self.config.audio.dtype == "bfloat16" else mx.float32
            )
        
        elif self.config.mode == ProcessingMode.CHUNKED:
            return self._model.transcribe(
                file_path,
                dtype=mx.bfloat16 if self.config.audio.dtype == "bfloat16" else mx.float32,
                chunk_duration=self.config.audio.chunk_duration,
                overlap_duration=self.config.audio.overlap_duration,
                chunk_callback=self.config.output.chunk_callback
            )
        
        else:
            raise ValueError(f"Unsupported processing mode for file: {self.config.mode}")
    
    def _transcribe_array(self, audio_data: mx.array) -> AlignedResult:
        logger.info("Transcribing audio array")
        
        mel = get_logmel(audio_data, self._model.preprocessor_config)
        
        results = self._model.generate(mel)
        result = results[0] if results else AlignedResult(text="", sentences=[], tokens=[])

        return self._apply_filler_filtering(result)
    
    def _transcribe_microphone(self) -> AlignedResult:
        logger.info("Transcribing from microphone")
        
        self._audio_source.start()
        
        try:
            duration = self.config.audio.chunk_duration or 5.0
            chunks = []
            start_time = time.time()
            
            for chunk in self._audio_source.get_audio_stream():
                chunks.append(chunk)
                if time.time() - start_time >= duration:
                    break
            
            if not chunks:
                return AlignedResult(text="", sentences=[], tokens=[])
            
            audio_data = mx.concatenate(chunks, axis=0)
            result = self._transcribe_array(audio_data)
            return self._apply_filler_filtering(result)
        
        finally:
            self._audio_source.stop()
    
    def transcribe_stream(
        self, 
        config: Optional[TranscriptionConfig] = None
    ) -> "StreamingTranscriber":
        if config is not None:
            stream_config = config
        else:
            stream_config = self.config
        
        return StreamingTranscriber(stream_config)
    
    async def transcribe_async(
        self,
        audio_source: Optional[Union[str, Path, mx.array]] = None,
        config: Optional[TranscriptionConfig] = None
    ) -> AlignedResult:

        # Run synchronous transcription in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.transcribe, 
            audio_source, 
            config
        )
    
    def get_performance_stats(self) -> dict:
        if self._performance_manager is None:
            return {}
        
        return self._performance_manager.get_memory_stats()
    
    def cleanup(self):
        if self._audio_source is not None:
            self._audio_source.stop()
        
        if self._performance_manager is not None:
            self._performance_manager.cleanup()
        
        if self._async_processor is not None:
            pass
        
        logger.info("Cleaned up UnifiedTranscriber resources")


class StreamingTranscriber:
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self._transcriber = None
        self._model = None
        self._streaming_context = None
        self._audio_source = None
        self._filler_filter = None
    
    def __enter__(self):
        dtype = mx.bfloat16 if self.config.model.dtype == "bfloat16" else mx.float32
        self._model = from_pretrained(self.config.model.model_name, dtype=dtype)
        
        self._streaming_context = self._model.transcribe_stream(
            context_size=self.config.model.context_size,
            depth=self.config.model.depth,
            keep_original_attention=self.config.model.keep_original_attention
        )
        self._streaming_context.__enter__()
        
        if self.config.audio.source != AudioSource.STREAM:
            self._audio_source = create_audio_source(
                self.config.audio,
                self._model.preprocessor_config
            )
            self._audio_source.start()

        if self.config.audio.filler_confidence_threshold > 0.0:
            filter_config = FillerFilterConfig(
                enabled=True,
                confidence_threshold=self.config.audio.filler_confidence_threshold,
                custom_filler_words=set(self.config.audio.custom_filler_words) if self.config.audio.custom_filler_words else None
            )
            self._filler_filter = FillerWordFilter(filter_config)

        logger.info("Started streaming transcription")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._audio_source is not None:
            self._audio_source.stop()
        
        if self._streaming_context is not None:
            self._streaming_context.__exit__(exc_type, exc_val, exc_tb)
        
        logger.info("Stopped streaming transcription")
    
    def add_audio(self, audio_data: mx.array):
        if self._streaming_context is None:
            raise RuntimeError("Streaming context not active")
        
        self._streaming_context.add_audio(audio_data)
    
    def get_audio_stream(self) -> Generator[mx.array, None, None]:
        if self._audio_source is None:
            raise RuntimeError("No audio source configured")
        
        return self._audio_source.get_audio_stream()
    
    @property
    def result(self) -> AlignedResult:
        if self._streaming_context is None:
            return AlignedResult(text="", sentences=[], tokens=[])

        result = self._streaming_context.result
        
        if self._filler_filter is not None:
            result = self._filler_filter.filter_result(result)

        return result
    
    @property
    def finalized_tokens(self):
        if self._streaming_context is None:
            return []
        
        return self._streaming_context.finalized_tokens
    
    @property
    def draft_tokens(self):
        if self._streaming_context is None:
            return []

        return self._streaming_context.draft_tokens

    def reset_buffers(self):
        if self._streaming_context is None:
            raise RuntimeError("Streaming context not active")

        self._streaming_context.reset_buffers()

        if (self._audio_source is not None and
            hasattr(self._audio_source, 'clear_audio_buffer')):
            self._audio_source.clear_audio_buffer()


def transcribe(
    audio_source: Union[str, Path, mx.array],
    **kwargs
) -> AlignedResult:

    config = TranscriptionConfig(
        model=ModelConfig(),
        **kwargs
    )
    
    transcriber = UnifiedTranscriber(config)
    return transcriber.transcribe(audio_source)


def transcribe_file(
    file_path: Union[str, Path],
    chunk_duration: Optional[float] = None,
    **kwargs
) -> AlignedResult:

    config = TranscriptionConfig.for_file_transcription(
        file_path=file_path,
        chunk_duration=chunk_duration,
        **kwargs
    )
    
    transcriber = UnifiedTranscriber(config)
    return transcriber.transcribe()


def transcribe_microphone(
    device: Optional[Union[int, str]] = None,
    duration: float = 5.0,
    **kwargs
) -> AlignedResult:

    config = TranscriptionConfig.for_realtime_transcription(
        microphone_device=device,
        **kwargs
    )
    config.audio.chunk_duration = duration
    
    transcriber = UnifiedTranscriber(config)
    return transcriber.transcribe()
