import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, Optional

import mlx.core as mx

from parakeet_mlx.audio.alignment import AlignedResult
from parakeet_mlx.core.config import TranscriptionConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    result: AlignedResult
    timestamp: float
    processing_time: float
    chunk_id: int


class AsyncAudioBuffer:
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._buffer = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._not_full = asyncio.Condition(self._lock)
    
    async def put(self, item: mx.array) -> bool:
        async with self._not_full:
            if len(self._buffer) >= self.max_size:
                self._buffer.popleft()
                logger.warning("Audio buffer full, dropping oldest chunk")
            
            self._buffer.append(item)
            self._not_empty.notify()
            return True
    
    async def get(self) -> Optional[mx.array]:
        async with self._not_empty:
            while not self._buffer:
                await self._not_empty.wait()
            
            return self._buffer.popleft()
    
    async def get_nowait(self) -> Optional[mx.array]:
        async with self._lock:
            if self._buffer:
                return self._buffer.popleft()
            return None
    
    def size(self) -> int:
        return len(self._buffer)
    
    def is_empty(self) -> bool:
        return len(self._buffer) == 0
    
    def is_full(self) -> bool:
        return len(self._buffer) >= self.max_size


class AsyncTranscriptionProcessor:
    
    def __init__(
        self,
        model: Any,
        config: TranscriptionConfig,
        result_callback: Optional[Callable[[ProcessingResult], None]] = None
    ):
        self.model = model
        self.config = config
        self.result_callback = result_callback

        self._is_running = False
        self._chunk_counter = 0
        self._processing_tasks = set()
        
        self._audio_buffer = AsyncAudioBuffer(
            max_size=config.performance.async_queue_size
        )
        self._result_queue = asyncio.Queue()
        
        self._processing_times = deque(maxlen=100)
        self._last_stats_time = time.time()
    
    async def start(self):
        if self._is_running:
            return
        
        self._is_running = True
        logger.info("Started async transcription processor")
        
        processing_task = asyncio.create_task(self._processing_loop())
        self._processing_tasks.add(processing_task)
        processing_task.add_done_callback(self._processing_tasks.discard)
    
    async def stop(self):
        if not self._is_running:
            return
        
        self._is_running = False
        
        for task in self._processing_tasks:
            task.cancel()
        
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        
        logger.info("Stopped async transcription processor")
    
    async def add_audio_chunk(self, audio_chunk: mx.array):
        if not self._is_running:
            await self.start()
        
        await self._audio_buffer.put(audio_chunk)
    
    async def get_result(self, timeout: Optional[float] = None) -> Optional[ProcessingResult]:
        try:
            if timeout is None:
                return await self._result_queue.get()
            else:
                return await asyncio.wait_for(self._result_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    async def _processing_loop(self):
        logger.info("Started processing loop")
        
        try:
            while self._is_running:
                audio_chunk = await self._audio_buffer.get()
                if audio_chunk is None:
                    continue
                
                task = asyncio.create_task(
                    self._process_chunk(audio_chunk, self._chunk_counter)
                )
                self._processing_tasks.add(task)
                task.add_done_callback(self._processing_tasks.discard)
                
                self._chunk_counter += 1
                
                if len(self._processing_tasks) > self.config.performance.async_queue_size:
                    done, pending = await asyncio.wait(
                        self._processing_tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    self._processing_tasks = pending
        
        except asyncio.CancelledError:
            logger.info("Processing loop cancelled")
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
    
    async def _process_chunk(self, audio_chunk: mx.array, chunk_id: int) -> ProcessingResult:
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._transcribe_chunk, 
                audio_chunk
            )
            
            processing_time = time.time() - start_time
            self._processing_times.append(processing_time)
            
            processing_result = ProcessingResult(
                result=result,
                timestamp=start_time,
                processing_time=processing_time,
                chunk_id=chunk_id
            )
            
            await self._result_queue.put(processing_result)
            
            if self.result_callback:
                try:
                    self.result_callback(processing_result)
                except Exception as e:
                    logger.error(f"Error in result callback: {e}")
            
            if time.time() - self._last_stats_time > 10.0:
                await self._log_performance_stats()
                self._last_stats_time = time.time()
            
            return processing_result
        
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")
            return ProcessingResult(
                result=AlignedResult(text="", sentences=[], tokens=[]),
                timestamp=start_time,
                processing_time=time.time() - start_time,
                chunk_id=chunk_id
            )
    
    def _transcribe_chunk(self, audio_chunk: mx.array) -> AlignedResult:
        from parakeet_mlx.audio import get_logmel
        
        mel = get_logmel(audio_chunk, self.model.preprocessor_config)
        
        results = self.model.generate(mel)
        return results[0] if results else AlignedResult(text="", sentences=[], tokens=[])
    
    async def _log_performance_stats(self):
        if not self._processing_times:
            return
        
        avg_time = sum(self._processing_times) / len(self._processing_times)
        min_time = min(self._processing_times)
        max_time = max(self._processing_times)
        
        logger.info(
            f"Performance stats - Avg: {avg_time:.3f}s, "
            f"Min: {min_time:.3f}s, Max: {max_time:.3f}s, "
            f"Buffer size: {self._audio_buffer.size()}, "
            f"Active tasks: {len(self._processing_tasks)}"
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        if not self._processing_times:
            return {}
        
        return {
            'avg_processing_time': sum(self._processing_times) / len(self._processing_times),
            'min_processing_time': min(self._processing_times),
            'max_processing_time': max(self._processing_times),
            'buffer_size': self._audio_buffer.size(),
            'active_tasks': len(self._processing_tasks),
            'total_chunks_processed': self._chunk_counter
        }


class AsyncStreamingTranscriber:
    
    def __init__(self, model: Any, config: TranscriptionConfig):
        self.model = model
        self.config = config
        self._processor = None
        self._streaming_context = None
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def start(self):
        self._streaming_context = self.model.transcribe_stream(
            context_size=self.config.model.context_size,
            depth=self.config.model.depth,
            keep_original_attention=self.config.model.keep_original_attention
        )
        self._streaming_context.__enter__()
        
        self._processor = AsyncTranscriptionProcessor(
            model=self._streaming_context,
            config=self.config
        )
        await self._processor.start()
        
        logger.info("Started async streaming transcriber")
    
    async def stop(self):
        if self._processor:
            await self._processor.stop()
            self._processor = None
        
        if self._streaming_context:
            self._streaming_context.__exit__(None, None, None)
            self._streaming_context = None
        
        logger.info("Stopped async streaming transcriber")
    
    async def add_audio(self, audio_data: mx.array):
        if self._processor is None:
            raise RuntimeError("Transcriber not started")
        
        await self._processor.add_audio_chunk(audio_data)
    
    async def get_result(self, timeout: Optional[float] = None) -> Optional[ProcessingResult]:
        if self._processor is None:
            raise RuntimeError("Transcriber not started")
        
        return await self._processor.get_result(timeout)
    
    async def transcribe_stream(
        self, 
        audio_stream: AsyncGenerator[mx.array, None]
    ) -> AsyncGenerator[ProcessingResult, None]:
        async for audio_chunk in audio_stream:
            await self.add_audio(audio_chunk)
            
            while True:
                result = await self.get_result(timeout=0.001)
                if result is None:
                    break
                yield result
    
    def get_current_result(self) -> Optional[AlignedResult]:
        if self._streaming_context is None:
            return None
        
        return self._streaming_context.result
