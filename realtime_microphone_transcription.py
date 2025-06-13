#!/usr/bin/env python

import argparse
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from parakeet_mlx.core.config import TranscriptionConfig
    from parakeet_mlx.core.transcriber import UnifiedTranscriber
except ImportError:
    print("Error: Could not import parakeet_mlx modules")
    print("Make sure you're running this from the parakeet-mlx-master directory")
    sys.exit(1)

def check_microphone_device(device_id):
    """Check if the specified microphone device is available"""
    try:
        import sounddevice as sd
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
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            for i, device in enumerate(input_devices):
                print(f"  {i}: {device['name']}")
        except Exception as e:
            print(f"Error listing devices: {e}")
        return

    # Check if the specified device is available
    if not check_microphone_device(args.device):
        print(f"Error: Microphone device {args.device} not available")
        print("Available input devices:")
        return

    # Create configuration with improved settings
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

    # Create transcriber
    transcriber = UnifiedTranscriber(config)

    # Set up graceful shutdown
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
        import traceback
        traceback.print_exc()
    finally:
        transcriber.cleanup()

def get_device_name(device_id):
    """Get the name of the audio device"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if device_id < len(input_devices):
            return input_devices[device_id]['name']
    except:
        pass
    return "Unknown"

if __name__ == "__main__":
    main()
