import asyncio
import logging
import multiprocessing
import platform
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import mlx.core as mx
import psutil

from parakeet_mlx.core.config import PerformanceConfig

logger = logging.getLogger(__name__)


class PerformanceManager:
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self._thread_pool = None
        self._gpu_devices = []
        self._memory_monitor = None
        
        self._setup_threading()
        self._setup_gpu()
        self._setup_memory_monitoring()
    
    def _setup_threading(self):
        if not self.config.enable_threading:
            return
        
        max_workers = self.config.max_workers
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"Initialized thread pool with {max_workers} workers")
    
    def _setup_gpu(self):
        if not self.config.enable_gpu:
            return
        
        try:
            available_devices = mx.metal.get_active_memory()
            if available_devices:
                if self.config.gpu_devices is None:
                    self._gpu_devices = list(range(len(available_devices)))
                else:
                    self._gpu_devices = [
                        i for i in self.config.gpu_devices 
                        if i < len(available_devices)
                    ]
                
                logger.info(f"Using GPU devices: {self._gpu_devices}")
            else:
                logger.warning("No GPU devices available")
        except Exception as e:
            logger.warning(f"Failed to setup GPU: {e}")
    
    def _setup_memory_monitoring(self):
        if not self.config.enable_memory_optimization:
            return
        
        self._memory_monitor = MemoryMonitor(
            max_buffer_size=self.config.max_buffer_size
        )
    
    def process_chunks_parallel(
        self, 
        chunks: List[Any], 
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        
        if not self.config.enable_threading or self._thread_pool is None:
            results = []
            for i, chunk in enumerate(chunks):
                result = process_func(chunk)
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, len(chunks))
            return results
        
        results = [None] * len(chunks)
        futures = {}
        
        for i, chunk in enumerate(chunks):
            future = self._thread_pool.submit(process_func, chunk)
            futures[future] = i
        
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(chunks))
            except Exception as e:
                logger.error(f"Error processing chunk {index}: {e}")
                results[index] = None
        
        return results
    
    async def process_chunks_async(
        self,
        chunks: List[Any],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        
        if not self.config.enable_async:
            return self.process_chunks_parallel(chunks, process_func, progress_callback)
        
        async def process_chunk_async(chunk, index):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, process_func, chunk)
        
        tasks = [
            process_chunk_async(chunk, i) 
            for i, chunk in enumerate(chunks)
        ]
        
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(chunks))
        
        return results
    
    def optimize_memory_usage(self, data: mx.array) -> mx.array:
        if self._memory_monitor is None:
            return data
        
        return self._memory_monitor.optimize_array(data)
    
    def get_optimal_batch_size(self, input_size: int) -> int:
        if self._memory_monitor is None:
            return 1
        
        return self._memory_monitor.get_optimal_batch_size(input_size)
    
    def cleanup(self):
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        
        if self._memory_monitor is not None:
            self._memory_monitor.cleanup()


class MemoryMonitor:
    
    def __init__(self, max_buffer_size: int):
        self.max_buffer_size = max_buffer_size
        self._memory_usage = {}
        self._lock = threading.Lock()
    
    def optimize_array(self, data: mx.array) -> mx.array:
        if data.size > self.max_buffer_size:
            logger.warning(f"Array size {data.size} exceeds max buffer size {self.max_buffer_size}")
        
        return data
    
    def get_optimal_batch_size(self, input_size: int) -> int:
        try:
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            
            memory_per_sample = input_size * 4 / (1024 * 1024)
            
            max_batch_size = int((available_mb * 0.5) / memory_per_sample)
            
            return max(1, min(max_batch_size, 32))
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}")
            return 1
    
    def get_memory_stats(self) -> Dict[str, float]:
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_percent': memory.percent,
                'free_gb': memory.free / (1024**3)
            }
        except Exception as e:
            logger.warning(f"Failed to get memory stats: {e}")
            return {}
    
    def cleanup(self):
        with self._lock:
            self._memory_usage.clear()


class GPULoadBalancer:
    
    def __init__(self, gpu_devices: List[int]):
        self.gpu_devices = gpu_devices
        self._current_device = 0
        self._lock = threading.Lock()
    
    def get_next_device(self) -> int:
        if not self.gpu_devices:
            return 0
        
        with self._lock:
            device = self.gpu_devices[self._current_device]
            self._current_device = (self._current_device + 1) % len(self.gpu_devices)
            return device
    
    def distribute_workload(self, workload: List[Any]) -> Dict[int, List[Any]]:
        if not self.gpu_devices:
            return {0: workload}
        
        distribution = {device: [] for device in self.gpu_devices}
        
        for i, item in enumerate(workload):
            device = self.gpu_devices[i % len(self.gpu_devices)]
            distribution[device].append(item)
        
        return distribution


def get_system_info() -> Dict[str, Any]:
    try:
        return {
            'cpu_count': multiprocessing.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory': psutil.virtual_memory()._asdict(),
            'gpu_available': mx.metal.get_active_memory() is not None,
            'platform': platform.system()
        }
    except Exception as e:
        logger.warning(f"Failed to get system info: {e}")
        return {}


def optimize_mlx_settings():
    try:
        mx.set_default_device(mx.gpu)
        logger.info("Optimized MLX settings for GPU usage")
    except Exception as e:
        logger.warning(f"Failed to optimize MLX settings: {e}")
