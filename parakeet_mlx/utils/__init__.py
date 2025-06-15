from parakeet_mlx.utils.device_manager import (
    find_audio_device,
    get_default_input_device,
    list_audio_devices,
    print_audio_devices,
    test_audio_device,
)
from parakeet_mlx.utils.error_handling import (
    AudioError,
    ConfigurationError,
    DeviceError,
    ModelError,
    PerformanceError,
    TranscriptionError,
    get_error_stats,
    handle_errors,
    reset_error_stats,
    safe_execute,
    setup_error_logging,
)
from parakeet_mlx.utils.model_loading import from_config, from_pretrained

__all__ = [
    # Model loading (backward compatibility)
    "from_pretrained",
    "from_config",

    # Device management
    "list_audio_devices",
    "get_default_input_device",
    "find_audio_device",
    "test_audio_device",
    "print_audio_devices",

    # Error handling
    "setup_error_logging",
    "get_error_stats",
    "reset_error_stats",
    "safe_execute",
    "handle_errors",
    "TranscriptionError",
    "AudioError",
    "ModelError",
    "ConfigurationError",
    "DeviceError",
    "PerformanceError",
]
