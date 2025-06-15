import logging
import time
from typing import Union

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    logger.warning("noisereduce library not available. Noise reduction will be disabled")


class NoiseReducer:
    def __init__(
        self,
        sample_rate: int = 16000,
        noise_reduction_strength: float = 0.3,
        enable_noise_reduction: bool = True
    ):
        self.sample_rate = sample_rate
        self.noise_reduction_strength = max(0.0, min(1.0, noise_reduction_strength))
        self.enable_noise_reduction = enable_noise_reduction and NOISEREDUCE_AVAILABLE
        
        self._total_processing_time = 0.0
        self._total_chunks_processed = 0
        
        self._nr_params = {
            'stationary': True,
            'prop_decrease': self.noise_reduction_strength,
            'n_fft': 512,
            'hop_length': 128,
            'time_mask_smooth_ms': 25,
            'freq_mask_smooth_hz': 250,
        }
        
        if not self.enable_noise_reduction:
            if not NOISEREDUCE_AVAILABLE:
                logger.info("Noise reduction disabled: noisereduce library not available")
            else:
                logger.info("Noise reduction disabled by configuration")
        else:
            logger.info(f"Noise reduction enabled with strength: {self.noise_reduction_strength}")
    
    def process_audio_chunk(self, audio_chunk: Union[mx.array, np.ndarray]) -> Union[mx.array, np.ndarray]:
        if not self.enable_noise_reduction or self.noise_reduction_strength == 0.0:
            return audio_chunk
        
        input_is_mlx = isinstance(audio_chunk, mx.array)
        
        try:
            start_time = time.perf_counter()
            
            if input_is_mlx:
                audio_np = np.array(audio_chunk)
            else:
                audio_np = audio_chunk
            
            if audio_np.ndim > 1:
                audio_np = audio_np.flatten()
            
            if len(audio_np) < self._nr_params['n_fft']:
                logger.debug(f"Skipping noise reduction: chunk too short ({len(audio_np)} samples)")
                return audio_chunk
            
            reduced_audio = nr.reduce_noise(
                y=audio_np,
                sr=self.sample_rate,
                **self._nr_params
            )
            
            if input_is_mlx:
                result = mx.array(reduced_audio)
            else:
                result = reduced_audio
            
            processing_time = time.perf_counter() - start_time
            self._total_processing_time += processing_time
            self._total_chunks_processed += 1

            if self._total_chunks_processed % 100 == 0:
                avg_time = self._total_processing_time / self._total_chunks_processed
                logger.debug(f"Noise reduction avg processing time: {avg_time*1000:.2f}ms per chunk")
            
            return result
            
        except Exception as e:
            logger.warning(f"Noise reduction failed, using original audio: {e}")
            return audio_chunk
    
    def update_strength(self, new_strength: float):
        old_strength = self.noise_reduction_strength
        self.noise_reduction_strength = max(0.0, min(1.0, new_strength))
        self._nr_params['prop_decrease'] = self.noise_reduction_strength
        
        logger.info(f"Noise reduction strength updated: {old_strength} -> {self.noise_reduction_strength}")
    
    def get_performance_stats(self) -> dict:
        if self._total_chunks_processed == 0:
            return {
                'enabled': self.enable_noise_reduction,
                'strength': self.noise_reduction_strength,
                'chunks_processed': 0,
                'avg_processing_time_ms': 0.0,
                'total_processing_time_s': 0.0
            }
        
        avg_time = self._total_processing_time / self._total_chunks_processed
        return {
            'enabled': self.enable_noise_reduction,
            'strength': self.noise_reduction_strength,
            'chunks_processed': self._total_chunks_processed,
            'avg_processing_time_ms': avg_time * 1000,
            'total_processing_time_s': self._total_processing_time
        }
    
    def reset_stats(self):
        self._total_processing_time = 0.0
        self._total_chunks_processed = 0
        logger.debug("Noise reduction performance stats reset")


def create_noise_reducer(
    sample_rate: int = 16000,
    noise_reduction_strength: float = 0.3,
    enable_noise_reduction: bool = True
) -> NoiseReducer:
    return NoiseReducer(
        sample_rate=sample_rate,
        noise_reduction_strength=noise_reduction_strength,
        enable_noise_reduction=enable_noise_reduction
    )


def test_noise_reduction(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    strength: float = 0.3
) -> np.ndarray:
    reducer = create_noise_reducer(
        sample_rate=sample_rate,
        noise_reduction_strength=strength,
        enable_noise_reduction=True
    )
    
    return reducer.process_audio_chunk(audio_data)
