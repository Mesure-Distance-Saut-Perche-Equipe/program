import logging
import os

LOGGING_LEVEL = int(os.getenv("LOGGING_LEVEL", logging.INFO))
LOG_FILE_NAME = os.getenv("LOG_FILE_NAME", "logs.log")

try:
    LOG_FILE_PATH = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), LOG_FILE_NAME
    )
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
except Exception as e:
    raise RuntimeError(f"Failed to configure log file path: {e}")

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(levelname)s::%(asctime)s::%(module)s::%(funcName)s::%(filename)s::%(lineno)d %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode="a"),
        logging.StreamHandler(),
    ],
    datefmt="%d-%b-%y %H:%M:%S",
)

logger = logging.getLogger(__name__)
