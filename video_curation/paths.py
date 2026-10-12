"""Data locations, read from the repo-root .env (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

# DATA_ROOT may be relative (to the repo root) or absolute
DATA_ROOT = (REPO_ROOT / os.getenv("DATA_ROOT", "data")).resolve()
METADATA_DIR = DATA_ROOT / "video_metadata"
VIDEO_DIR = DATA_ROOT / "videomedia"
LOGS_DIR = DATA_ROOT / "logs"
COMPANY_DIR = DATA_ROOT / "company_name"


def batch_csv(batch: str) -> Path:
    """'batch4' or 'batch4.csv' -> COMPANY_DIR/batch4.csv"""
    return COMPANY_DIR / (batch if batch.lower().endswith(".csv") else f"{batch}.csv")


def remote() -> tuple[str, str]:
    """(REMOTE_HOST, REMOTE_DIR) for uploads to the cluster."""
    host, path = os.getenv("REMOTE_HOST"), os.getenv("REMOTE_DIR")
    if not host or not path:
        raise RuntimeError("Set REMOTE_HOST and REMOTE_DIR in .env")
    return host, path
