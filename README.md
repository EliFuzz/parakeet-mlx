# 🦜 Parakeet MLX

> **Next-Gen ASR on Apple Silicon** - A high-performance, production-ready speech recognition engine leveraging Apple's MLX framework for lightning-fast inference on M-series chips. Built with modern Python, featuring real-time streaming, advanced audio processing, and high performance monitoring.

Parakeet MLX brings state-of-the-art automatic speech recognition to Apple Silicon devices, offering blazing-fast inference with minimal memory footprint. Built on Apple's MLX framework, it delivers production-ready ASR with real-time streaming capabilities.

## ✨ Key Features

- 🚀 **Apple Silicon Optimized**: Native MLX implementation for M1/M2/M3 chips
- 🎤 **Real-time Transcription**: Live microphone input with streaming support
- 📁 **Flexible Input**: File transcription with chunking for long audio
- 🔧 **Advanced Audio Processing**: Built-in noise reduction and silence detection
- 🎛️ **Unified API**: Simple interface with extensive configuration options
- 📊 **Performance Monitoring**: Built-in profiling and optimization tools
- 🔄 **Streaming Context**: Maintains context across audio chunks
- 🎧 **Noise Reduction**: Built-in dynamic noise reduction and silence detection
- 🔁 **Filler Filtering**: Built-in filler word filtering

## 🚀 Quick Start

### Basic Usage

```python
from parakeet_mlx import transcribe_file, transcribe_microphone, UnifiedTranscriber

# Transcribe an audio file
result = transcribe_file("audio.wav")
print(result.text)

# Real-time microphone transcription
result = transcribe_microphone(duration=5.0)
print(result.text)

# Streaming transcription
transcriber = UnifiedTranscriber()
with transcriber.transcribe_stream() as stream:
    for audio_chunk in stream.get_audio_stream():
        stream.add_audio(audio_chunk)
        if stream.result:
            print(stream.result.text)
```

### File Transcription Example

```python
from parakeet_mlx import UnifiedTranscriber, TranscriptionConfig

# Configure for file transcription
config = TranscriptionConfig.for_file_transcription(
    "audio.wav",
    chunk_duration=30.0,
    include_timestamps=True
)

transcriber = UnifiedTranscriber(config)
result = transcriber.transcribe()

print(f"Transcription: {result.text}")
print(f"Duration: {result.duration:.2f}s")
print(f"Confidence: {result.confidence:.2f}")
```

### Real-time Transcription Example

```python
from parakeet_mlx import UnifiedTranscriber, TranscriptionConfig

# Configure for real-time transcription
config = TranscriptionConfig.for_realtime_transcription(
    microphone_device=0,
    silence_duration_ms=200,
    noise_reduction=0.3
)

transcriber = UnifiedTranscriber(config)

# Start streaming transcription
print("🎤 Listening... (Ctrl+C to stop)")
with transcriber.transcribe_stream() as stream:
    for audio_chunk in stream.get_audio_stream():
        stream.add_audio(audio_chunk)
        if stream.result and stream.result.text.strip():
            print(f">> {stream.result.text}")
            stream.reset_buffers()
```

### Complete Example of Real-time Transcription

```python
import argparse
import signal
import sys
from parakeet_mlx.core.config import TranscriptionConfig
from parakeet_mlx.core.transcriber import UnifiedTranscriber
import sounddevice as sd
import traceback


def check_microphone_device(device_id):
    try:
        
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]


        if device_id >= len(input_devices):
            for i, device in enumerate(input_devices):
                print(f"  {i}: {device['name']}")
            return False

        device = input_devices[device_id]
        return True
    except Exception as e:
        print(f"Warning: Could not check audio devices: {e}")
        print(f"Continuing with device ID {device_id}..")
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Real-time microphone transcription using Parakeet MLX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python realtime_microphone_transcription.py                    # Use default settings
  python realtime_microphone_transcription.py --list-devices     # List available microphones
  python realtime_microphone_transcription.py --device 0         # Use specific microphone
  python realtime_microphone_transcription.py --silence-threshold 0.05  # Adjust sensitivity
        """
    )
    parser.add_argument("--device", type=int, default=1, help="Microphone device ID (default: 1)")
    parser.add_argument("--silence-threshold", type=float, default=0.8, help="Silence threshold - lower = more sensitive (default: 0.8)")
    parser.add_argument("--silence-duration", type=int, default=100, help="Silence duration in ms before processing (default: 100)")
    parser.add_argument("--noise-reduction", type=float, default=0.3, help="Noise reduction strength 0.0-1.0 (default: 0.3)")
    parser.add_argument("--min-length", type=int, default=3, help="Minimum text length to display (default: 3)")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.list_devices:
        print("Available input devices:")
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        for i, device in enumerate(input_devices):
            print(f"  {i}: {device['name']}")

    if not check_microphone_device(args.device):
        print(f"Error: Microphone device {args.device} not available")
        print("Available input devices:")
        return

    config = TranscriptionConfig.for_realtime_transcription(
        microphone_device=args.device,
        silence_duration_ms=args.silence_duration,
        silence_threshold=args.silence_threshold,
        noise_reduction=args.noise_reduction,
        filler_confidence_threshold=0.2
    )

    if args.verbose:
        config.log_level = "DEBUG"
        config.verbose = True

    transcriber = UnifiedTranscriber(config)

    def signal_handler(_sig, _frame):
        print("\nShutting down..")
        transcriber.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    print(f"Starting transcription with device {args.device}: {get_device_name(args.device)}")
    print("Speak into the microphone. Press Ctrl+C to stop.\n")

    try:
        with transcriber.transcribe_stream() as stream:
            print("Listening... (speak now)")
            for audio_chunk in stream.get_audio_stream():
                stream.add_audio(audio_chunk)
                result = stream.result
                if result and result.text and result.text.strip():
                    print(result.text.strip())
                    stream.reset_buffers()

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        transcriber.cleanup()

def get_device_name(device_id):
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if device_id < len(input_devices):
            return input_devices[device_id]['name']
    except:
        pass
    return "Unknown"

if __name__ == "__main__":
    main()
```