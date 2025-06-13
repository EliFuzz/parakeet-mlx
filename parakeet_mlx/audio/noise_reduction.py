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
    """
    Real-time noise reduction processor optimized for transcription pipelines
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        noise_reduction_strength: float = 0.3,
        enable_noise_reduction: bool = True
    ):
        """
        Initialize the noise reducer
        
        Args:
            sample_rate: Audio sample rate in Hz
            noise_reduction_strength: Noise reduction strength (0.0-1.0: 0 - No noise reduction)
            enable_noise_reduction: Whether to enable noise reduction
        """
        self.sample_rate = sample_rate
        self.noise_reduction_strength = max(0.0, min(1.0, noise_reduction_strength))
        self.enable_noise_reduction = enable_noise_reduction and NOISEREDUCE_AVAILABLE
        
        # Performance tracking
        self._total_processing_time = 0.0
        self._total_chunks_processed = 0
        
        # Noise reduction parameters optimized for real-time performance
        self._nr_params = {
            'stationary': True,  # Use stationary for better real-time performance
            'prop_decrease': self.noise_reduction_strength,
            'n_fft': 512,  # Smaller FFT for lower latency
            'hop_length': 128,  # Smaller hop for better temporal resolution
            'time_mask_smooth_ms': 25,  # Reduced smoothing for lower latency
            'freq_mask_smooth_hz': 250,  # Reduced smoothing for lower latency
        }
        
        if not self.enable_noise_reduction:
            if not NOISEREDUCE_AVAILABLE:
                logger.info("Noise reduction disabled: noisereduce library not available")
            else:
                logger.info("Noise reduction disabled by configuration")
        else:
            logger.info(f"Noise reduction enabled with strength: {self.noise_reduction_strength}")
    
    def process_audio_chunk(self, audio_chunk: Union[mx.array, np.ndarray]) -> Union[mx.array, np.ndarray]:
        """
        Apply noise reduction to an audio chunk
        
        Args:
            audio_chunk: Input audio chunk as MLX array or numpy array
            
        Returns:
            Processed audio chunk with noise reduction applied
        """
        if not self.enable_noise_reduction or self.noise_reduction_strength == 0.0:
            return audio_chunk
        
        # Track input type for consistent output
        input_is_mlx = isinstance(audio_chunk, mx.array)
        
        try:
            start_time = time.perf_counter()
            
            # Convert to numpy for noisereduce processing
            if input_is_mlx:
                audio_np = np.array(audio_chunk)
            else:
                audio_np = audio_chunk
            
            # Ensure audio is 1D
            if audio_np.ndim > 1:
                audio_np = audio_np.flatten()
            
            # Skip processing if chunk is too short
            if len(audio_np) < self._nr_params['n_fft']:
                logger.debug(f"Skipping noise reduction: chunk too short ({len(audio_np)} samples)")
                return audio_chunk
            
            # Apply noise reduction
            reduced_audio = nr.reduce_noise(
                y=audio_np,
                sr=self.sample_rate,
                **self._nr_params
            )
            
            # Convert back to original format
            if input_is_mlx:
                result = mx.array(reduced_audio)
            else:
                result = reduced_audio
            
            # Update performance metrics
            processing_time = time.perf_counter() - start_time
            self._total_processing_time += processing_time
            self._total_chunks_processed += 1
            
            # Log performance occasionally
            if self._total_chunks_processed % 100 == 0:
                avg_time = self._total_processing_time / self._total_chunks_processed
                logger.debug(f"Noise reduction avg processing time: {avg_time*1000:.2f}ms per chunk")
            
            return result
            
        except Exception as e:
            logger.warning(f"Noise reduction failed, using original audio: {e}")
            return audio_chunk
    
    def update_strength(self, new_strength: float):
        """
        Update noise reduction strength dynamically
        
        Args:
            new_strength: New noise reduction strength (0.0-1.0)
        """
        old_strength = self.noise_reduction_strength
        self.noise_reduction_strength = max(0.0, min(1.0, new_strength))
        self._nr_params['prop_decrease'] = self.noise_reduction_strength
        
        logger.info(f"Noise reduction strength updated: {old_strength} -> {self.noise_reduction_strength}")
    
    def get_performance_stats(self) -> dict:
        """
        Get performance statistics for noise reduction processing
        
        Returns:
            Dictionary with performance metrics
        """
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
        """Reset performance statistics"""
        self._total_processing_time = 0.0
        self._total_chunks_processed = 0
        logger.debug("Noise reduction performance stats reset")


def create_noise_reducer(
    sample_rate: int = 16000,
    noise_reduction_strength: float = 0.3,
    enable_noise_reduction: bool = True
) -> NoiseReducer:
    """
    Factory function to create a noise reducer instance
    
    Args:
        sample_rate: Audio sample rate in Hz
        noise_reduction_strength: Noise reduction strength (0.0-1.0)
        enable_noise_reduction: Whether to enable noise reduction
        
    Returns:
        NoiseReducer instance
    """
    return NoiseReducer(
        sample_rate=sample_rate,
        noise_reduction_strength=noise_reduction_strength,
        enable_noise_reduction=enable_noise_reduction
    )


# Utility function for testing noise reduction
def test_noise_reduction(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    strength: float = 0.3
) -> np.ndarray:
    """
    Test noise reduction on audio data
    
    Args:
        audio_data: Input audio as numpy array
        sample_rate: Sample rate in Hz
        strength: Noise reduction strength (0.0-1.0)
        
    Returns:
        Processed audio with noise reduction applied
    """
    reducer = create_noise_reducer(
        sample_rate=sample_rate,
        noise_reduction_strength=strength,
        enable_noise_reduction=True
    )
    
    return reducer.process_audio_chunk(audio_data)
