"""
logger_config.py
================
Centralized, production-grade logging infrastructure for the EDXSO School
Influencer Discovery Tool.
Supports:
  - Console output with color formatting
  - Persistent rotating log file (data/system_activity.log)
  - In-memory ring buffer (LogBufferHandler) for real-time Streamlit UI display
"""

import logging
import os
from collections import deque
from datetime import datetime
from typing import List, Dict, Any

# In-memory ring buffer for UI display
LOG_BUFFER_SIZE = 100
_log_buffer: deque = deque(maxlen=LOG_BUFFER_SIZE)

LOG_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "system_activity.log"
)


class RingBufferHandler(logging.Handler):
    """Logging handler that appends log records to an in-memory deque."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3],
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
            }
            _log_buffer.append(entry)
        except Exception:
            self.handleError(record)


def setup_logger(name: str = "edxso_discovery") -> logging.Logger:
    """Configure and return a structured logger instance."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 2. File Handler
    try:
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
        fh = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        logger.warning("Could not initialize file log handler: %s", e)

    # 3. In-memory Buffer Handler for UI
    bh = RingBufferHandler()
    bh.setLevel(logging.INFO)
    logger.addHandler(bh)

    return logger


def get_recent_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve the most recent log entries from the in-memory ring buffer."""
    return list(_log_buffer)[-limit:]


def clear_logs() -> None:
    """Clear in-memory ring buffer."""
    _log_buffer.clear()
