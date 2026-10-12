"""
Central configuration loader for the pair-extraction pipeline.

Reads config.yaml (sitting next to this file) and exposes:
  - CONFIG: the fully-resolved nested dict (paths templated to absolute paths).
  - Module-level constants (OUTPUT_BASE_DIR, DATA_CSV, AUTOSHOT_MODEL_PATH,
    SHOT_THRESHOLD, ...) kept compatible with the old s1_config.py so the stage
    scripts only need `import config` instead of `import s1_config as config`.

Path templating: values may contain {data_root}, {version}, {output_dir};
these are resolved in dependency order below.
"""

import os
import yaml

# Directory containing this file (the package dir). Used so model paths resolve
# regardless of the current working directory.
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PKG_DIR, "config.yaml")


def _resolve_paths(paths: dict) -> dict:
    """Resolve {data_root}/{version}/{output_dir} templating in dependency order."""
    data_root = paths["data_root"]
    version = str(paths["version"])
    output_dir = paths["output_dir"].format(data_root=data_root, version=version)

    ctx = {"data_root": data_root, "version": version, "output_dir": output_dir}

    resolved = dict(paths)
    resolved["output_dir"] = output_dir
    for key, val in paths.items():
        if key in ("data_root", "version", "output_dir", "default_batch", "batch_csvs"):
            continue
        if isinstance(val, str):
            resolved[key] = val.format(**ctx)

    # Resolve the per-batch CSV map too.
    resolved["batch_csvs"] = {
        name: tmpl.format(**ctx) for name, tmpl in paths.get("batch_csvs", {}).items()
    }
    return resolved


def load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = dict(raw)
    cfg["paths"] = _resolve_paths(raw["paths"])
    # Make the model checkpoint path absolute relative to the package dir.
    model = dict(raw.get("model", {}))
    ckpt = model.get("autoshot_ckpt", "")
    if ckpt and not os.path.isabs(ckpt):
        model["autoshot_ckpt"] = os.path.join(PKG_DIR, ckpt)
    cfg["model"] = model
    return cfg


CONFIG = load_config()

_paths = CONFIG["paths"]
_model = CONFIG["model"]
_params = CONFIG["params"]

# ---------------------------------------------------------------------------
# Backwards-compatible constants (mirror the old s1_config.py names)
# ---------------------------------------------------------------------------
VIDEO_BASE_DIR = _paths["video_dir"]
OUTPUT_BASE_DIR = _paths["output_dir"]
SHOTSPLITS_DIR = _paths["shotsplits_dir"]
CLUSTERS_DIR = _paths["clusters_dir"]
EMBEDDINGS_DIR = _paths["embeddings_dir"]
KEYFRAMES_DIR = _paths["keyframes_dir"]
STATUS_FILE = _paths["status_file"]
FULLMATCH_OUT = _paths["fullmatch_out"]

DEFAULT_BATCH = _paths.get("default_batch")
BATCH_CSVS = _paths["batch_csvs"]
DATA_CSV = BATCH_CSVS.get(DEFAULT_BATCH) if DEFAULT_BATCH else None

# AutoShot model
AUTOSHOT_MODEL_PATH = _model["autoshot_ckpt"]

# Shot detection parameters
SHOT_THRESHOLD = _params["shot_threshold"]
SHOT_WINDOW_BATCH_SIZE = _params.get("shot_window_batch_size", 64)
MIN_GAP_FRAMES = _params["min_gap_frames"]
MIN_FRAMES_IN_SHOT = _params["min_frames_in_shot"]

# Keyframe extraction parameters
FRAME_WIDTH = _params["frame_width"]
FRAME_HEIGHT = _params["frame_height"]
KEYFRAME_PERCENTILES = _params["keyframe_percentiles"]

# Image quality
IMAGE_QUALITY = _params["image_quality"]
IMAGE_SUBSAMPLING = _params["image_subsampling"]

# Embedding parameters
EMBEDDING_BATCH_SIZE = _params["embedding_batch_size"]
EMBEDDING_DIM = _params["embedding_dim"]

# Processing
NUM_WORKERS = _params["num_workers"]

# Pairing keep-gate
MIN_MAPPING_DEGREE = _params["min_mapping_degree"]

# Environment
FFMPEG_PATH = _model["ffmpeg_path"]
os.environ["PATH"] = f"{os.path.dirname(FFMPEG_PATH)}:" + os.environ.get("PATH", "")
