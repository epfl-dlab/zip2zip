import logging
import os
import sys


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[94m",  # Blue
        logging.INFO: "\033[92m",  # Green
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",  # Red
        logging.CRITICAL: "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        level_color = self.COLORS.get(record.levelno, "")
        levelname = (
            f"{level_color}{record.levelname}{self.RESET}"
            if level_color
            else record.levelname
        )
        record.levelname = levelname
        return super().format(record)


def configure_logging():
    log_level_str = os.getenv("ZIP2ZIP_LOGLEVEL", "WARNING").upper()
    log_level = getattr(logging, log_level_str, logging.WARNING)

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    logger = logging.getLogger(__name__)
    logger.info("Logger initialized at %s level", log_level_str)
