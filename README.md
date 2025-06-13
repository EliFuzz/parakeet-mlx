# 🦜 Parakeet MLX

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-0.22.1+-green.svg)](https://github.com/ml-explore/mlx)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://badge.fury.io/py/parakeet-mlx.svg)](https://badge.fury.io/py/parakeet-mlx)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Optimized-orange.svg)](https://developer.apple.com/silicon/)

> **A unified, high-performance implementation of NVIDIA's Parakeet ASR models for Apple Silicon using MLX with real-time transcription capabilities.**

Parakeet MLX brings state-of-the-art automatic speech recognition to Apple Silicon devices, offering blazing-fast inference with minimal memory footprint. Built on Apple's MLX framework, it delivers production-ready ASR with real-time streaming capabilities.

## ✨ Key Features

- 🚀 **Apple Silicon Optimized**: Native MLX implementation for M1/M2/M3 chips
- 🎯 **Multiple Model Architectures**: TDT, RNN-T, CTC, and hybrid models
- 🎤 **Real-time Transcription**: Live microphone input with streaming support
- 📁 **Flexible Input**: File transcription with chunking for long audio
- 🔧 **Advanced Audio Processing**: Built-in noise reduction and silence detection
- 🎛️ **Unified API**: Simple interface with extensive configuration options
- 📊 **Performance Monitoring**: Built-in profiling and optimization tools
- 🔄 **Streaming Context**: Maintains context across audio chunks

## 🚀 Quick Start

### Installation

```bash
# install from source
git clone https://github.com/EliFuzz/parakeet-mlx.git
cd parakeet-mlx
pip install -e .
```

### Basic Usage

```python
import parakeet_mlx as px

# Transcribe an audio file
result = px.transcribe_file("audio.wav")
print(result.text)

# Real-time microphone transcription
result = px.transcribe_microphone(duration=5.0)
print(result.text)

# Streaming transcription
transcriber = px.UnifiedTranscriber()
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

## 🎯 Supported Models

| Model | Architecture | Size | Description |
|-------|-------------|------|-------------|
| `mlx-community/parakeet-tdt-0.6b-v2` | TDT | 600M | Default model, balanced accuracy/speed |
| `mlx-community/parakeet-rnnt-1.1b` | RNN-T | 1.1B | High accuracy RNN-T model |
| `mlx-community/parakeet-ctc-0.6b` | CTC | 600M | Fast CTC model for batch processing |
| `mlx-community/parakeet-tdt-ctc-1.1b` | Hybrid | 1.1B | Combined TDT-CTC architecture |

## 🛠️ Command Line Interface

```bash
# Basic file transcription
parakeet-mlx transcribe audio.wav

# Real-time transcription
parakeet-mlx-realtime

# List available microphones
parakeet-mlx-realtime --list-devices
```

## 📚 Documentation

For detailed documentation, examples, and advanced usage patterns, visit our **[GitHub Wiki](https://github.com/EliFuzz/parakeet-mlx/wiki)**:

### 📖 Getting Started
- **[Quick Start Guide](https://github.com/EliFuzz/parakeet-mlx/wiki/Quick-Start)** - Installation and basic usage
- **[Examples](https://github.com/EliFuzz/parakeet-mlx/wiki/Examples)** - Comprehensive usage examples and demonstrations
- **[CLI Reference](https://github.com/EliFuzz/parakeet-mlx/wiki/CLI-Reference)** - Command-line interface documentation

### 🔧 Technical Documentation
- **[API Reference](https://github.com/EliFuzz/parakeet-mlx/wiki/API-Reference)** - Complete API documentation
- **[Configuration Guide](https://github.com/EliFuzz/parakeet-mlx/wiki/Configuration-Guide)** - Detailed configuration options
- **[Architecture Overview](https://github.com/EliFuzz/parakeet-mlx/wiki/Architecture-Overview)** - Core architecture and design patterns

### 🚀 Advanced Topics
- **[Performance Optimization](https://github.com/EliFuzz/parakeet-mlx/wiki/Performance-Optimization)** - Performance tuning and benchmarks
- **[Advanced Usage](https://github.com/EliFuzz/parakeet-mlx/wiki/Advanced-Usage)** - Complex integration patterns and best practices
- **[Troubleshooting](https://github.com/EliFuzz/parakeet-mlx/wiki/Troubleshooting)** - Common issues and solutions

### 👨‍💻 Development
- **[Development Guide](https://github.com/EliFuzz/parakeet-mlx/wiki/Development-Guide)** - Contributing and development setup

## 🤝 Contributing

We welcome contributions! Please see our **[Development Guide](https://github.com/EliFuzz/parakeet-mlx/wiki/Development-Guide)** for details on:

- Setting up the development environment
- Code style and standards
- Testing procedures
- Submitting pull requests

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NVIDIA** for the original Parakeet ASR models
- **Apple** for the MLX framework
- **Hugging Face** for model hosting and distribution
- The open-source community for contributions and feedback

---

<div align="center">

**Made with ❤️ for the AI/ML community**

[⭐ Star this repo](https://github.com/EliFuzz/parakeet-mlx) • [📚 Documentation](https://github.com/EliFuzz/parakeet-mlx/wiki) • [🐛 Report Bug](https://github.com/EliFuzz/parakeet-mlx/issues) • [💡 Request Feature](https://github.com/EliFuzz/parakeet-mlx/issues)

</div>
