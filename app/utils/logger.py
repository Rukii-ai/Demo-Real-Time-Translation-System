"""Logger for the translation system."""

# Python's built-in logging module handles log messages, levels, and formatting.
import logging

# Return a named logger so each part of the app can write messages in a consistent way.
def get_logger(name: str) -> logging.Logger:
    # Get (or create) the logger for this name.
    logger = logging.getLogger(name)
    # INFO shows normal progress, warnings, and errors without too much noise.
    logger.setLevel(logging.INFO)

    # Only add a handler once so the same message does not print multiple times.
    if not logger.handlers:
        # StreamHandler sends logs to the console/terminal.
        handler = logging.StreamHandler()
        # Format each log line so it includes time, logger name, level, and message.
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # Attach the format to the handler.
        handler.setFormatter(formatter)
        # Connect the handler to the logger so messages actually appear.
        logger.addHandler(handler)

    # Hand the configured logger back to the caller.
    return logger