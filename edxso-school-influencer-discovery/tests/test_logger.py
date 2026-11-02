"""
test_logger.py
==============
Unit tests for structured logging and ring buffer collection.
"""

from logger_config import setup_logger, get_recent_logs, clear_logs


def test_logger_ring_buffer():
    clear_logs()
    test_logger = setup_logger("test_module")
    test_logger.info("Test message alpha")
    test_logger.warning("Test message beta")

    logs = get_recent_logs(10)
    assert len(logs) >= 2
    messages = [l["message"] for l in logs]
    assert "Test message alpha" in messages
    assert "Test message beta" in messages
    assert logs[-1]["level"] in ["INFO", "WARNING"]


def test_clear_logs():
    test_logger = setup_logger("test_module_clear")
    test_logger.info("Message before clear")
    clear_logs()
    assert len(get_recent_logs(10)) == 0
