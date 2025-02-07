import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_path(key):
    val = os.getenv(key)
    assert val is not None
    return Path(val)


RESEARCH_DATA_PATH = load_path("RESEARCH_DATA_PATH")
CACHE_PATH = load_path("CACHE_PATH")
RESULTS_PATH = load_path("RESULTS_PATH")
TASKS_PATH = load_path("TASKS_PATH")
