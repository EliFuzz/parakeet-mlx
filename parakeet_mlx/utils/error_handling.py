import functools
import logging
import sys
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TranscriptionError(Exception):
    """Base exception for transcription errors"""
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation"""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'severity': self.severity.value,
            'error_code': self.error_code,
            'details': self.details
        }


class AudioError(TranscriptionError):
    """Audio-related errors"""
    pass


class ModelError(TranscriptionError):
    """Model-related errors"""
    pass


class ConfigurationError(TranscriptionError):
    """Configuration-related errors"""
    pass


class DeviceError(TranscriptionError):
    """Audio device-related errors"""
    pass


class PerformanceError(TranscriptionError):
    """Performance-related errors"""
    pass


class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self, logger_name: str = "parakeet_mlx"):
        self.logger = logging.getLogger(logger_name)
        self._error_counts = {}
        self._recovery_strategies = {}
        
        # Setup default recovery strategies
        self._setup_default_recovery_strategies()
    
    def _setup_default_recovery_strategies(self):
        """Setup default error recovery strategies"""
        
        def retry_with_fallback(func, *args, **kwargs):
            """Retry with fallback parameters"""
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Retrying with fallback parameters: {e}")
                # Implement fallback logic here
                raise
        
        def graceful_degradation(func, *args, **kwargs):
            """Gracefully degrade functionality"""
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Graceful degradation: {e}")
                # Return minimal/default result
                return None
        
        self._recovery_strategies = {
            AudioError: retry_with_fallback,
            DeviceError: graceful_degradation,
            PerformanceError: graceful_degradation
        }
    
    def handle_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None,
        raise_on_critical: bool = True
    ) -> Optional[Any]:
        """
        Handle an error with appropriate logging and recovery.
        
        Args:
            error: The exception to handle
            context: Additional context information
            raise_on_critical: Whether to re-raise critical errors
        
        Returns:
            Recovery result or None
        """
        context = context or {}
        
        # Determine error type and severity
        if isinstance(error, TranscriptionError):
            error_type = type(error)
            severity = error.severity
        else:
            error_type = type(error)
            severity = self._determine_severity(error)
        
        # Log the error
        self._log_error(error, severity, context)
        
        # Track error counts
        error_key = f"{error_type.__name__}:{str(error)[:100]}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        # Apply recovery strategy if available
        recovery_strategy = self._recovery_strategies.get(error_type)
        if recovery_strategy and severity != ErrorSeverity.CRITICAL:
            try:
                return recovery_strategy(lambda: None)  # Placeholder
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")
        
        # Re-raise critical errors or if no recovery is possible
        if severity == ErrorSeverity.CRITICAL and raise_on_critical:
            raise error
        
        return None
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type"""
        
        # Critical errors that should stop execution
        critical_errors = (
            SystemExit, KeyboardInterrupt, MemoryError,
            ImportError, ModuleNotFoundError
        )
        
        # High severity errors
        high_errors = (
            FileNotFoundError, PermissionError, OSError,
            ValueError, TypeError
        )
        
        # Medium severity errors
        medium_errors = (
            RuntimeError, ConnectionError, TimeoutError
        )
        
        if isinstance(error, critical_errors):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, high_errors):
            return ErrorSeverity.HIGH
        elif isinstance(error, medium_errors):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _log_error(
        self, 
        error: Exception, 
        severity: ErrorSeverity, 
        context: Dict[str, Any]
    ):
        """Log error with appropriate level and formatting"""
        
        error_info = {
            'error_type': type(error).__name__,
            'message': str(error),
            'severity': severity.value,
            'context': context
        }
        
        # Add traceback for high/critical errors
        if severity in (ErrorSeverity.HIGH, ErrorSeverity.CRITICAL):
            error_info['traceback'] = traceback.format_exc()
        
        # Log with appropriate level
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical error: {error_info}")
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(f"High severity error: {error_info}")
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Medium severity error: {error_info}")
        else:
            self.logger.info(f"Low severity error: {error_info}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            'total_errors': sum(self._error_counts.values()),
            'error_counts': self._error_counts.copy(),
            'most_common_errors': sorted(
                self._error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
    
    def reset_stats(self):
        """Reset error statistics"""
        self._error_counts.clear()


# Global error handler instance
_error_handler = ErrorHandler()


def handle_errors(
    error_types: Union[Type[Exception], tuple] = Exception,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    raise_on_critical: bool = True,
    context: Optional[Dict[str, Any]] = None
):
    """
    Decorator for automatic error handling.
    
    Args:
        error_types: Exception types to handle
        severity: Default severity for untyped errors
        raise_on_critical: Whether to re-raise critical errors
        context: Additional context information
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                # Add function context
                func_context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # Truncate for logging
                    'kwargs': str(kwargs)[:200]
                }
                if context:
                    func_context.update(context)
                
                return _error_handler.handle_error(
                    e, 
                    context=func_context,
                    raise_on_critical=raise_on_critical
                )
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    error_types: Union[Type[Exception], tuple] = Exception,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Default return value on error
        error_types: Exception types to catch
        context: Additional context information
        **kwargs: Function keyword arguments
    
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except error_types as e:
        func_context = {
            'function': func.__name__,
            'default_return': str(default_return)
        }
        if context:
            func_context.update(context)
        
        _error_handler.handle_error(e, context=func_context, raise_on_critical=False)
        return default_return


def validate_audio_file(file_path: Union[str, Path]) -> Path:
    """
    Validate audio file exists and is readable
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Validated Path object
    
    Raises:
        AudioError: If file is invalid
    """
    path = Path(file_path)
    
    if not path.exists():
        raise AudioError(
            f"Audio file not found: {path}",
            severity=ErrorSeverity.HIGH,
            error_code="FILE_NOT_FOUND",
            details={'file_path': str(path)}
        )
    
    if not path.is_file():
        raise AudioError(
            f"Path is not a file: {path}",
            severity=ErrorSeverity.HIGH,
            error_code="NOT_A_FILE",
            details={'file_path': str(path)}
        )
    
    # Check file size (basic validation)
    try:
        size = path.stat().st_size
        if size == 0:
            raise AudioError(
                f"Audio file is empty: {path}",
                severity=ErrorSeverity.HIGH,
                error_code="EMPTY_FILE",
                details={'file_path': str(path)}
            )
        
        # Warn about very large files (>1GB)
        if size > 1024 * 1024 * 1024:
            _error_handler.logger.warning(
                f"Large audio file detected: {path} ({size / (1024**3):.1f} GB)"
            )
    
    except OSError as e:
        raise AudioError(
            f"Cannot access audio file: {path}",
            severity=ErrorSeverity.HIGH,
            error_code="ACCESS_ERROR",
            details={'file_path': str(path), 'os_error': str(e)}
        )
    
    return path


def validate_model_name(model_name: str) -> str:
    """
    Validate model name format
    
    Args:
        model_name: Model name to validate
    
    Returns:
        Validated model name
    
    Raises:
        ModelError: If model name is invalid
    """
    if not model_name or not isinstance(model_name, str):
        raise ModelError(
            "Model name must be a non-empty string",
            severity=ErrorSeverity.HIGH,
            error_code="INVALID_MODEL_NAME"
        )
    
    # Basic format validation for HuggingFace model names
    if '/' in model_name and len(model_name.split('/')) == 2:
        return model_name
    elif model_name.startswith('mlx-community/'):
        return model_name
    else:
        _error_handler.logger.warning(
            f"Model name format may be invalid: {model_name}"
        )
        return model_name


def setup_error_logging(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    verbose: bool = False
):
    """
    Setup error logging configuration
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        verbose: Enable verbose logging
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    logging.getLogger().addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG if verbose else getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)
    
    # Set verbose mode
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def get_error_stats() -> Dict[str, Any]:
    """Get global error statistics"""
    return _error_handler.get_error_stats()


def reset_error_stats():
    """Reset global error statistics"""
    _error_handler.reset_stats()
