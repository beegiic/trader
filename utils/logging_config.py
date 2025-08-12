import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

import structlog
from structlog.typing import FilteringBoundLogger


def configure_logging(
    log_level: str = "INFO",
    log_file: str = "trading_bot.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> FilteringBoundLogger:
    """Configure structured logging with JSON format, trace IDs, and file rotation."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file_path = log_dir / log_file
    
    # Configure timestamper
    timestamper = structlog.processors.TimeStamper(fmt="ISO")
    
    # Shared processors for both console and file
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    
    # Configure console output (human-readable)
    console_processors = shared_processors + [
        structlog.dev.ConsoleRenderer(colors=True)
    ]
    
    # Configure file output (JSON)
    file_processors = shared_processors + [
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer()
    ]
    
    # Configure standard library logging
    import logging
    import logging.handlers
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Root logger configuration
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[file_handler, console_handler],
    )
    
    # Configure structlog
    structlog.configure(
        processors=file_processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Create logger with trace ID context
    logger = structlog.get_logger()
    
    return logger


def get_trace_id() -> str:
    """Generate a unique trace ID for request tracking."""
    return str(uuid.uuid4())[:8]


def bind_trace_id(logger: FilteringBoundLogger, trace_id: str = None) -> FilteringBoundLogger:
    """Bind a trace ID to the logger for request tracking."""
    if trace_id is None:
        trace_id = get_trace_id()
    return logger.bind(trace_id=trace_id)


class LoggerMixin:
    """Mixin class to provide logging capabilities to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = structlog.get_logger(self.__class__.__name__)
        self._trace_id = get_trace_id()
    
    @property
    def logger(self) -> FilteringBoundLogger:
        """Get logger with bound trace ID."""
        return self._logger.bind(trace_id=self._trace_id)
    
    def new_trace(self) -> str:
        """Generate new trace ID for this instance."""
        self._trace_id = get_trace_id()
        return self._trace_id


def log_function_call(func_name: str, **kwargs) -> Dict[str, Any]:
    """Helper to create consistent function call log entries."""
    return {
        "event": "function_call",
        "function": func_name,
        **kwargs
    }


def log_trade_event(event_type: str, symbol: str, **kwargs) -> Dict[str, Any]:
    """Helper to create consistent trade event log entries."""
    return {
        "event": "trade_event",
        "event_type": event_type,
        "symbol": symbol,
        **kwargs
    }


def log_risk_event(event_type: str, **kwargs) -> Dict[str, Any]:
    """Helper to create consistent risk event log entries."""
    return {
        "event": "risk_event",
        "event_type": event_type,
        **kwargs
    }


# Global logger instance
logger: FilteringBoundLogger = None


def init_logging():
    """Initialize global logging configuration."""
    global logger
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "trading_bot.log")
    logger = configure_logging(log_level=log_level, log_file=log_file)
    logger.info("Logging initialized", level=log_level, file=log_file)


def get_logger(name: str = None) -> FilteringBoundLogger:
    """Get a logger instance with optional name binding."""
    if logger is None:
        init_logging()
    
    if name:
        return logger.bind(component=name)
    return logger