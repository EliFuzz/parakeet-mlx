import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class AudioSource(str, Enum):
    """Supported audio input sources"""
    FILE = "file"
    MICROPHONE = "microphone"
    STREAM = "stream"


class OutputFormat(str, Enum):
    """Supported output formats"""
    TEXT = "text"
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"
    ALL = "all"


class ProcessingMode(str, Enum):
    """Processing modes for different use cases"""
    BATCH = "batch"          # Process entire file at once
    CHUNKED = "chunked"      # Process in chunks for long files
    STREAMING = "streaming"  # Real-time streaming with context
    REALTIME = "realtime"    # Live microphone input


class AudioConfig(BaseModel):
    """Audio input and processing configuration"""

    # Input source configuration
    source: AudioSource = AudioSource.FILE
    file_path: Optional[Path] = None
    microphone_device: Optional[Union[int, str]] = None  # Device ID or name
    sample_rate: int = 16000
    channels: int = 1

    # Audio processing parameters
    chunk_duration: Optional[float] = None  # Seconds, None = no chunking
    overlap_duration: float = 15.0  # Seconds
    buffer_size: int = 1024  # Samples for real-time processing

    # Silence detection parameters
    enable_silence_detection: bool = False  # Enable dynamic silence detection
    silence_duration_ms: int = 200  # Milliseconds of silence to trigger transcription
    silence_threshold: float = 0.01  # Amplitude threshold for silence detection
    min_speech_duration_ms: int = 100  # Minimum speech duration before processing
    max_buffer_duration_s: float = 30.0  # Maximum buffer duration before forced processing

    # Audio quality settings
    dtype: str = "bfloat16"  # "bfloat16" or "float32"

    # Noise reduction settings
    noise_reduction: float = 0.3  # Noise reduction strength (0.0-1.0)

    # Filler word filtering settings
    filler_confidence_threshold: float = 0.8  # Confidence threshold for filler detection
    custom_filler_words: Optional[List[str]] = None  # Custom filler words to filter

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
        # Only validate file existence if source is FILE and not a dummy/placeholder path
        if (v is not None and
            values.get('source') == AudioSource.FILE and
            str(v) != "dummy.wav" and
            not Path(v).exists()):
            raise ValueError(f"Audio file not found: {v}")
        return v


class ModelConfig(BaseModel):
    """Model configuration and parameters"""
    
    # Model selection
    model_name: str = "mlx-community/parakeet-tdt-0.6b-v2"
    model_type: Optional[str] = None  # Auto-detected from config
    
    # Processing parameters
    context_size: tuple[int, int] = (256, 256)  # (left, right) for streaming
    depth: int = 1  # Cache depth for streaming
    keep_original_attention: bool = False
    
    # Performance settings
    dtype: str = "bfloat16"
    batch_size: int = 1
    
    @validator('dtype')
    def validate_dtype(cls, v):
        if v not in ["bfloat16", "float32"]:
            raise ValueError("dtype must be 'bfloat16' or 'float32'")
        return v


class PerformanceConfig(BaseModel):
    """Performance optimization configuration"""
    
    # Multi-threading
    enable_threading: bool = True
    max_workers: Optional[int] = None  # None = auto-detect
    
    # GPU settings
    enable_gpu: bool = True
    gpu_devices: Optional[List[int]] = None  # None = use all available
    gpu_memory_limit: Optional[float] = None  # GB, None = no limit
    
    # Memory management
    enable_memory_optimization: bool = True
    max_buffer_size: int = 1024 * 1024  # Samples
    
    # Async processing
    enable_async: bool = True
    async_queue_size: int = 10


class OutputConfig(BaseModel):
    """Output configuration and formatting"""
    
    # Output format
    format: OutputFormat = OutputFormat.TEXT
    output_dir: Optional[Path] = None
    output_template: str = "{filename}"
    
    # Text processing
    highlight_words: bool = False
    include_timestamps: bool = True
    include_confidence: bool = False
    
    # Filtering
    min_confidence: float = 0.0
    filter_silence: bool = True
    
    # Callbacks
    chunk_callback: Optional[Callable] = None
    progress_callback: Optional[Callable] = None


class TranscriptionConfig(BaseModel):
    """Unified configuration for all transcription modes"""
    
    # Core configurations
    audio: AudioConfig = Field(default_factory=AudioConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    # Processing mode (auto-detected if not specified)
    mode: Optional[ProcessingMode] = None
    
    # Logging
    log_level: str = "INFO"
    verbose: bool = False
    
    def __post_init__(self):
        """Auto-detect processing mode if not specified"""
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
        """Create configuration for file transcription"""
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
        """Create configuration for real-time microphone transcription"""
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
        """Create configuration for streaming transcription"""
        audio_config = AudioConfig(source=AudioSource.STREAM)
        model_config = ModelConfig(context_size=context_size)
        return cls(audio=audio_config, model=model_config, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.dict()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TranscriptionConfig":
        """Create configuration from dictionary"""
        # Handle nested configurations properly
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
        """Setup logging based on configuration"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)


# Default configurations for common use cases
def get_default_file_config():
    """Get default file transcription configuration"""
    return TranscriptionConfig.for_file_transcription("dummy.wav")

def get_default_realtime_config():
    """Get default real-time transcription configuration"""
    return TranscriptionConfig.for_realtime_transcription()

def get_default_streaming_config():
    """Get default streaming transcription configuration"""
    return TranscriptionConfig.for_streaming_transcription()
