
from parakeet_mlx.audio.alignment import AlignedResult, AlignedSentence, AlignedToken
from parakeet_mlx.models.parakeet import DecodingConfig, ParakeetTDT, ParakeetTDTArgs
from parakeet_mlx.utils.model_loading import from_pretrained

from parakeet_mlx.core.transcriber import (
    UnifiedTranscriber,
    StreamingTranscriber,
    transcribe,
    transcribe_file,
    transcribe_microphone
)
from parakeet_mlx.core.config import (
    TranscriptionConfig,
    AudioConfig,
    ModelConfig,
    PerformanceConfig,
    OutputConfig,
    AudioSource,
    OutputFormat,
    ProcessingMode
)

from parakeet_mlx.utils.device_manager import (
    list_audio_devices,
    get_default_input_device,
    find_audio_device
)

__all__ = [
    # Legacy API (backward compatibility)
    "DecodingConfig",
    "ParakeetTDTArgs",
    "ParakeetTDT",
    "from_pretrained",
    "AlignedResult",
    "AlignedSentence",
    "AlignedToken",

    # New unified interface
    "UnifiedTranscriber",
    "StreamingTranscriber",
    "transcribe",
    "transcribe_file",
    "transcribe_microphone",

    # Configuration
    "TranscriptionConfig",
    "AudioConfig",
    "ModelConfig",
    "PerformanceConfig",
    "OutputConfig",
    "AudioSource",
    "OutputFormat",
    "ProcessingMode",

    # Utilities
    "list_audio_devices",
    "get_default_input_device",
    "find_audio_device",
]
