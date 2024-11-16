import logging
import os.path

LOGGING_LEVEL = logging.INFO

LOG_FILE_NAME = "logs.log"
LOG_FILE_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), LOG_FILE_NAME
)

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(levelname)s::%(asctime)s::%(module)s::%(funcName)s::%(filename)s::%(lineno)d %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH, mode="a")],
    datefmt="%d-%b-%y %H:%M:%S",
)

logger = logging.getLogger(__name__)
