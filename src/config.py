"""Paths and tunable constants shared by all pipeline stages."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", ROOT / "Dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", ROOT / "work"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", ROOT / "output"))

N_JOBS = max(1, (os.cpu_count() or 4) - 2)
SEED = 42

# Candidate generation: per S2/S3 record, how many S1 candidates each channel keeps,
# and the final cap after merging channels.
K_NAME_TFIDF = 10
K_NAME_TOKEN = 10
K_ADDR = 10
K_FINAL = 8
# Blocking keys that occur in more than this many S1 records are too generic to use.
MAX_KEY_DF = 200


def split_dir(split: str) -> Path:
    return DATA_DIR / split


def source_path(split: str, src: int) -> Path:
    return split_dir(split) / f"{split}_source{src}.tsv"


def work_path(split: str, name: str) -> Path:
    d = WORK_DIR / split
    d.mkdir(parents=True, exist_ok=True)
    return d / name
