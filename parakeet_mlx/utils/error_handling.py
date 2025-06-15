import functools
import logging
import sys
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TranscriptionError(Exception):
    
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
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'severity': self.severity.value,
            'error_code': self.error_code,
            'details': self.details
        }


class AudioError(TranscriptionError):
    pass


class ModelError(TranscriptionError):
    pass


class ConfigurationError(TranscriptionError):
    pass


class DeviceError(TranscriptionError):
    pass


class PerformanceError(TranscriptionError):
    pass


class ErrorHandler:
    
    def __init__(self, logger_name: str = "parakeet_mlx"):
        self.logger = logging.getLogger(logger_name)
        self._error_counts = {}
        self._recovery_strategies = {}
        
        self._setup_default_recovery_strategies()
    
    def _setup_default_recovery_strategies(self):
        
        def retry_with_fallback(func, *args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Retrying with fallback parameters: {e}")
                raise
        
        def graceful_degradation(func, *args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Graceful degradation: {e}")
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
        context = context or {}
        
        if isinstance(error, TranscriptionError):
            error_type = type(error)
            severity = error.severity
        else:
            error_type = type(error)
            severity = self._determine_severity(error)
        
        self._log_error(error, severity, context)
        
        error_key = f"{error_type.__name__}:{str(error)[:100]}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        recovery_strategy = self._recovery_strategies.get(error_type)
        if recovery_strategy and severity != ErrorSeverity.CRITICAL:
            try:
                return recovery_strategy(lambda: None)  # Placeholder
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")
        
        if severity == ErrorSeverity.CRITICAL and raise_on_critical:
            raise error
        
        return None
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        
        critical_errors = (
            SystemExit, KeyboardInterrupt, MemoryError,
            ImportError, ModuleNotFoundError
        )
        
        high_errors = (
            FileNotFoundError, PermissionError, OSError,
            ValueError, TypeError
        )
        
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
        
        error_info = {
            'error_type': type(error).__name__,
            'message': str(error),
            'severity': severity.value,
            'context': context
        }
        
        if severity in (ErrorSeverity.HIGH, ErrorSeverity.CRITICAL):
            error_info['traceback'] = traceback.format_exc()
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical error: {error_info}")
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(f"High severity error: {error_info}")
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Medium severity error: {error_info}")
        else:
            self.logger.info(f"Low severity error: {error_info}")
    
    def get_error_stats(self) -> Dict[str, Any]:
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
        self._error_counts.clear()


_error_handler = ErrorHandler()


def handle_errors(
    error_types: Union[Type[Exception], tuple] = Exception,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    raise_on_critical: bool = True,
    context: Optional[Dict[str, Any]] = None
):

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
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
    
    try:
        size = path.stat().st_size
        if size == 0:
            raise AudioError(
                f"Audio file is empty: {path}",
                severity=ErrorSeverity.HIGH,
                error_code="EMPTY_FILE",
                details={'file_path': str(path)}
            )
        
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

    if not model_name or not isinstance(model_name, str):
        raise ModelError(
            "Model name must be a non-empty string",
            severity=ErrorSeverity.HIGH,
            error_code="INVALID_MODEL_NAME"
        )
    
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

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[]
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    logging.getLogger().addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG if verbose else getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def get_error_stats() -> Dict[str, Any]:
    return _error_handler.get_error_stats()


def reset_error_stats():
    _error_handler.reset_stats()
