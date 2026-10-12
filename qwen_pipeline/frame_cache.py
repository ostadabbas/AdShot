#!/usr/bin/env python3
"""
frame_cache.py — persistent per-shot frame cache keyed by focal_ad_id.

The eval/training input is k frames per shot, extracted via ffmpeg
(build_message.extract_kframes_per_shot). That extraction is the per-row
bottleneck (~5-37s vs ~2-10s for generate). This caches frames on disk so
re-runs and multi-fps sweeps don't re-extract.

Layout:
  data/frames/<focal_ad_id>/k<fps>/shot_NN/frame_MM.jpg

get_frame_uris(...) returns (frame_uris_by_shot, shot_durations) exactly like
extract_kframes_per_shot, but skips ffmpeg when a complete frame set is present.
shot_durations are computed without decoding (same math as build_message).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path("/projects/adsum-mllm-om")
EVAL_DIR = PROJECT_ROOT / "mllm" / "evaluations"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from build_message import extract_kframes_per_shot  # noqa: E402
from data_prep import shot_durations as _durations, _as_list  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE / "data" / "frames"


def _shot_dir(cache_root: Path, focal_ad_id: str, fps: int, shot_i: int) -> Path:
    return cache_root / str(focal_ad_id) / f"k{fps}" / f"shot_{shot_i:02d}"


def _cached_uris(cache_root: Path, focal_ad_id: str, fps: int, n_shots: int):
    """Return frame URIs if a complete cached set exists, else None."""
    by_shot: List[List[str]] = []
    for i in range(1, n_shots + 1):
        d = _shot_dir(cache_root, focal_ad_id, fps, i)
        # Match only final frames (frame_01.jpg ...), not frame_tmp_*.jpg leftovers.
        frames = sorted(d.glob("frame_[0-9][0-9].jpg"))
        if len(frames) != fps:          # incomplete / missing -> cache miss
            return None
        by_shot.append([f.resolve().as_uri() for f in frames])
    return by_shot


def get_frame_uris(
    video_path: Path,
    video_info: Dict[str, Any],
    fps: int,
    cache_root: Path = DEFAULT_CACHE,
) -> Tuple[List[List[str]], List[float]]:
    """Cache-aware version of extract_kframes_per_shot."""
    focal_ad_id = video_info.get("focal_ad_id") or Path(video_path).stem
    shotsplit = _as_list(video_info["focal_shotsplit"])
    n_shots = len(shotsplit) if shotsplit and shotsplit[0] == 0 else len(shotsplit)
    durs = _durations(shotsplit, video_info["focal_ad_length"])
    n_shots = len(durs)

    hit = _cached_uris(cache_root, focal_ad_id, fps, n_shots)
    if hit is not None:
        return hit, durs

    # Miss: extract into the cache dir for this ad, then read back as URIs.
    out_dir = cache_root / str(focal_ad_id) / f"k{fps}"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_shot, durs = extract_kframes_per_shot(video_path, video_info, output_dir=out_dir, k=fps)
    return by_shot, durs


if __name__ == "__main__":
    # Smoke test on a single fullmatch row.
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "data" / "fullmatch_all.csv"))
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    info = df.loc[args.row].to_dict()
    vp = (PROJECT_ROOT / info["focal_relative_path"]).resolve()
    import time
    t0 = time.time(); uris, durs = get_frame_uris(vp, info, args.fps); t1 = time.time()
    t2 = time.time(); uris2, _ = get_frame_uris(vp, info, args.fps); t3 = time.time()
    print(f"shots={len(uris)} frames/shot={[len(u) for u in uris][:5]}...")
    print(f"miss(extract)={t1-t0:.2f}s  hit(cache)={t3-t2:.3f}s  durs[:3]={[round(d,2) for d in durs[:3]]}")
    assert uris == uris2, "cache returned different URIs than extraction"
    print("OK: cache consistent")
