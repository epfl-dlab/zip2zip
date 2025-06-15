from zip2zip.logging_utils import configure_logging

configure_logging()

# Now logging is active
import logging

logger = logging.getLogger("zip2zip")
logger.debug("This will show if ZIP2ZIP_LOGLEVEL=DEBUG")
