import logging
import platform
from typing import Dict, List, Optional, Union

import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioDevice:
    
    def __init__(self, device_id: int, name: str, channels: int, sample_rate: float):
        self.id = device_id
        self.name = name
        self.channels = channels
        self.sample_rate = sample_rate
    
    def __str__(self) -> str:
        return f"Device {self.id}: {self.name} ({self.channels} channels, {self.sample_rate} Hz)"
    
    def __repr__(self) -> str:
        return f"AudioDevice(id={self.id}, name='{self.name}', channels={self.channels}, sample_rate={self.sample_rate})"
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'channels': self.channels,
            'sample_rate': self.sample_rate
        }


class DeviceManager:
    
    def __init__(self):
        self._devices = None
        self._default_device = None
        self._refresh_devices()
    
    def _refresh_devices(self):
        try:
            devices = sd.query_devices()
            self._devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    audio_device = AudioDevice(
                        device_id=i,
                        name=device['name'],
                        channels=device['max_input_channels'],
                        sample_rate=device['default_samplerate']
                    )
                    self._devices.append(audio_device)
            
            try:
                default_device_info = sd.query_devices(kind='input')
                default_id = sd.default.device[0] if sd.default.device[0] is not None else 0
                
                self._default_device = AudioDevice(
                    device_id=default_id,
                    name=default_device_info['name'],
                    channels=default_device_info['max_input_channels'],
                    sample_rate=default_device_info['default_samplerate']
                )
            except Exception as e:
                logger.warning(f"Could not determine default device: {e}")
                if self._devices:
                    self._default_device = self._devices[0]
        
        except Exception as e:
            logger.error(f"Failed to refresh audio devices: {e}")
            self._devices = []
            self._default_device = None
    
    def list_devices(self, refresh: bool = False) -> List[AudioDevice]:
        if refresh or self._devices is None:
            self._refresh_devices()
        
        return self._devices.copy() if self._devices else []
    
    def get_default_device(self, refresh: bool = False) -> Optional[AudioDevice]:
        if refresh or self._default_device is None:
            self._refresh_devices()
        
        return self._default_device
    
    def get_device_by_id(self, device_id: int, refresh: bool = False) -> Optional[AudioDevice]:
        devices = self.list_devices(refresh)
        
        for device in devices:
            if device.id == device_id:
                return device
        
        return None
    
    def get_device_by_name(self, name: str, refresh: bool = False) -> Optional[AudioDevice]:
        devices = self.list_devices(refresh)
        name_lower = name.lower()
        
        for device in devices:
            if device.name.lower() == name_lower:
                return device
        
        for device in devices:
            if name_lower in device.name.lower():
                return device
        
        return None
    
    def find_device(self, identifier: Union[int, str], refresh: bool = False) -> Optional[AudioDevice]:
        if isinstance(identifier, int):
            return self.get_device_by_id(identifier, refresh)
        elif isinstance(identifier, str):
            return self.get_device_by_name(identifier, refresh)
        else:
            raise ValueError(f"Invalid identifier type: {type(identifier)}")
    
    def test_device(self, device: Union[AudioDevice, int], duration: float = 1.0) -> bool:
        device_id = device.id if isinstance(device, AudioDevice) else device
        
        try:
            recording = sd.rec(
                int(duration * 16000),  # Assume 16kHz sample rate
                samplerate=16000,
                channels=1,
                device=device_id,
                dtype='float32'
            )
            sd.wait()
            
            if recording is not None and len(recording) > 0:
                return True
            else:
                return False
        
        except Exception as e:
            logger.warning(f"Device test failed for device {device_id}: {e}")
            return False
    
    def get_system_info(self) -> Dict:
        try:
            return {
                'platform': platform.system(),
                'sounddevice_version': sd.__version__,
                'default_input_device': self.get_default_device().to_dict() if self.get_default_device() else None,
                'total_input_devices': len(self.list_devices()),
                'available_sample_rates': [8000, 16000, 22050, 44100, 48000],  # Common rates
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {}


_device_manager = DeviceManager()


def list_audio_devices(refresh: bool = False) -> List[AudioDevice]:
    return _device_manager.list_devices(refresh)


def get_default_input_device(refresh: bool = False) -> Optional[AudioDevice]:
    return _device_manager.get_default_device(refresh)


def find_audio_device(identifier: Union[int, str], refresh: bool = False) -> Optional[AudioDevice]:

    return _device_manager.find_device(identifier, refresh)


def test_audio_device(device: Union[AudioDevice, int], duration: float = 1.0) -> bool:

    return _device_manager.test_device(device, duration)


def print_audio_devices():
    devices = list_audio_devices(refresh=True)
    default_device = get_default_input_device()
    
    print("Available Audio Input Devices:")
    print("=" * 50)
    
    if not devices:
        print("No audio input devices found")
        return
    
    for device in devices:
        status = " (DEFAULT)" if default_device and device.id == default_device.id else ""
        print(f"{device}{status}")
    
    print(f"\nTotal devices: {len(devices)}")


def get_recommended_device() -> Optional[AudioDevice]:

    devices = list_audio_devices()
    
    if not devices:
        return None
    
    def device_score(device: AudioDevice) -> float:
        score = 0.0
        
        if device.sample_rate >= 44100:
            score += 2.0
        elif device.sample_rate >= 16000:
            score += 1.0
        
        if device.channels >= 2:
            score += 1.0
        
        name_lower = device.name.lower()
        if any(keyword in name_lower for keyword in ['usb', 'external', 'microphone']):
            score += 1.0
        
        return score
    
    scored_devices = [(device, device_score(device)) for device in devices]
    scored_devices.sort(key=lambda x: x[1], reverse=True)
    
    return scored_devices[0][0] if scored_devices else None
