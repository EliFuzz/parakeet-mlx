import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class AudioSource(str, Enum):
    FILE = "file"
    MICROPHONE = "microphone"
    STREAM = "stream"


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"
    ALL = "all"


class ProcessingMode(str, Enum):
    BATCH = "batch"
    CHUNKED = "chunked"
    STREAMING = "streaming"
    REALTIME = "realtime"


class AudioConfig(BaseModel):

    source: AudioSource = AudioSource.FILE
    file_path: Optional[Path] = None
    microphone_device: Optional[Union[int, str]] = None
    sample_rate: int = 16000
    channels: int = 1

    chunk_duration: Optional[float] = None
    overlap_duration: float = 15.0
    buffer_size: int = 1024

    enable_silence_detection: bool = False
    silence_duration_ms: int = 200
    silence_threshold: float = 0.01
    min_speech_duration_ms: int = 100
    max_buffer_duration_s: float = 30.0

    dtype: str = "bfloat16"

    noise_reduction: float = 0.3

    filler_confidence_threshold: float = 0.8
    custom_filler_words: Optional[List[str]] = None

    @validator('dtype')
    def validate_dtype(cls, v):
        if v not in ["bfloat16", "float32"]:
            raise ValueError("dtype must be 'bfloat16' or 'float32'")
        return v

    @validator('noise_reduction')
    def validate_noise_reduction(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("noise_reduction must be between 0.0 and 1.0")
        return v

    @validator('filler_confidence_threshold')
    def validate_filler_confidence_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("filler_confidence_threshold must be between 0.0 and 1.0")
        return v
    
    @validator('file_path')
    def validate_file_path(cls, v, values):
        if values.get('source') == AudioSource.FILE and v is None:
            raise ValueError("file_path is required when source is FILE")
        if (v is not None and
            values.get('source') == AudioSource.FILE and
            str(v) != "dummy.wav" and
            not Path(v).exists()):
            raise ValueError(f"Audio file not found: {v}")
        return v


class ModelConfig(BaseModel):
    
    model_name: str = "mlx-community/parakeet-tdt-0.6b-v2"
    model_type: Optional[str] = None
    
    context_size: tuple[int, int] = (256, 256)
    depth: int = 1
    keep_original_attention: bool = False
    
    dtype: str = "bfloat16"
    batch_size: int = 1
    
    @validator('dtype')
    def validate_dtype(cls, v):
        if v not in ["bfloat16", "float32"]:
            raise ValueError("dtype must be 'bfloat16' or 'float32'")
        return v


class PerformanceConfig(BaseModel):
    
    enable_threading: bool = True
    max_workers: Optional[int] = None
    
    enable_gpu: bool = True
    gpu_devices: Optional[List[int]] = None
    gpu_memory_limit: Optional[float] = None
    
    enable_memory_optimization: bool = True
    max_buffer_size: int = 1024 * 1024
    
    enable_async: bool = True
    async_queue_size: int = 10


class OutputConfig(BaseModel):
    
    format: OutputFormat = OutputFormat.TEXT
    output_dir: Optional[Path] = None
    output_template: str = "{filename}"
    
    highlight_words: bool = False
    include_timestamps: bool = True
    include_confidence: bool = False
    
    min_confidence: float = 0.0
    filter_silence: bool = True
    
    chunk_callback: Optional[Callable] = None
    progress_callback: Optional[Callable] = None


class TranscriptionConfig(BaseModel):
    
    audio: AudioConfig = Field(default_factory=AudioConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    mode: Optional[ProcessingMode] = None
    
    log_level: str = "INFO"
    verbose: bool = False
    
    def __post_init__(self):
        if self.mode is None:
            if self.audio.source == AudioSource.FILE:
                if self.audio.chunk_duration is not None:
                    self.mode = ProcessingMode.CHUNKED
                else:
                    self.mode = ProcessingMode.BATCH
            elif self.audio.source == AudioSource.MICROPHONE:
                self.mode = ProcessingMode.REALTIME
            elif self.audio.source == AudioSource.STREAM:
                self.mode = ProcessingMode.STREAMING
    
    @classmethod
    def for_file_transcription(
        cls,
        file_path: Union[str, Path],
        chunk_duration: Optional[float] = None,
        **kwargs
    ) -> "TranscriptionConfig":
        audio_config = AudioConfig(
            source=AudioSource.FILE,
            file_path=Path(file_path),
            chunk_duration=chunk_duration
        )
        return cls(audio=audio_config, **kwargs)
    
    @classmethod
    def for_realtime_transcription(
        cls,
        microphone_device: Optional[Union[int, str]] = None,
        enable_silence_detection: bool = True,
        silence_duration_ms: int = 200,
        noise_reduction: float = 0.8,
        filler_confidence_threshold: int = 0.2,
        **kwargs
    ) -> "TranscriptionConfig":
        audio_config = AudioConfig(
            source=AudioSource.MICROPHONE,
            microphone_device=microphone_device,
            enable_silence_detection=enable_silence_detection,
            silence_duration_ms=silence_duration_ms,
            noise_reduction=noise_reduction,
            filler_confidence_threshold=filler_confidence_threshold
        )
        return cls(audio=audio_config, **kwargs)
    
    @classmethod
    def for_streaming_transcription(
        cls,
        context_size: tuple[int, int] = (256, 256),
        **kwargs
    ) -> "TranscriptionConfig":
        audio_config = AudioConfig(source=AudioSource.STREAM)
        model_config = ModelConfig(context_size=context_size)
        return cls(audio=audio_config, model=model_config, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TranscriptionConfig":
        """Create configuration from dictionary"""
        if 'audio' in config_dict and isinstance(config_dict['audio'], dict):
            config_dict['audio'] = AudioConfig(**config_dict['audio'])
        if 'model' in config_dict and isinstance(config_dict['model'], dict):
            config_dict['model'] = ModelConfig(**config_dict['model'])
        if 'performance' in config_dict and isinstance(config_dict['performance'], dict):
            config_dict['performance'] = PerformanceConfig(**config_dict['performance'])
        if 'output' in config_dict and isinstance(config_dict['output'], dict):
            config_dict['output'] = OutputConfig(**config_dict['output'])

        return cls(**config_dict)
    
    def setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)


def get_default_file_config():
    return TranscriptionConfig.for_file_transcription("dummy.wav")

def get_default_realtime_config():
    return TranscriptionConfig.for_realtime_transcription()

def get_default_streaming_config():
    return TranscriptionConfig.for_streaming_transcription()
