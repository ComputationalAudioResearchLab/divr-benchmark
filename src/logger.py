import logging
from pathlib import Path


class Logger:
    def __init__(self, log_path: str, key: str):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        __log_path = Path(f"{log_path}/{key}.log")
        __log_path.parent.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(
            "%(asctime)s:%(levelname)s : %(name)s : %(message)s"
        )
        file_handler = logging.FileHandler(__log_path)
        file_handler.setFormatter(formatter)
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        self.info = root_logger.info
        self.debug = root_logger.debug
        self.warn = root_logger.warn
        self.error = root_logger.error
