import logging
import os


def configure_logging():
    log_level_str = os.getenv("ZIP2ZIP_LOGLEVEL", "WARNING").upper()
    log_level = getattr(logging, log_level_str, logging.WARNING)

    # Force reconfiguration by removing existing handlers
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=log_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info("Logger initialized at %s level", log_level_str)
