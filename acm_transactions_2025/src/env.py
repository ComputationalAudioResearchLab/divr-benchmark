import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_path(key):
    val = os.getenv(key)
    assert val is not None
    return Path(val)


RESEARCH_DATA_PATH = load_path("RESEARCH_DATA_PATH")
