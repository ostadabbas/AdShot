#!/usr/bin/env python3
"""
kf_build.py — pack per-shot keyframes into ONE HDF5 file per fps.

Why: writing ~300k tiny JPEGs to NFS is metadata-bound and starves the GPU.
One HDF5 container turns that into a single big-file sequential write, and
becomes a reusable keyframe artifact every eval/training run reads.

Design (single-writer): a worker Pool does the CPU-heavy extract->decode->resize
and streams numpy arrays back to the MAIN process, which is the only HDF5 writer
(plain h5py has no safe concurrent write). Worker temp frames go to node-local
/tmp, never NFS. Already-extracted JPEGs under data/frames are reused as-is
(no re-extraction); anything missing is extracted fresh to local tmp.

Frames are stored DECODED (uint8) but resized to the eval budget via Qwen's
patch-aligned smart_resize(max_pixels), so stored pixels == what the model sees
(storing native resolution would balloon the file). Frames of one video share
H,W, so each ad is one dense dataset:

  /<focal_ad_id>  uint8 (n_shots, k, H, W, 3)   attrs: shot_durations, relative_path

Run (human):
  /projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python keyframes/kf_build.py \
    --csv data/fullmatch_all.csv --fps 5 \
    --out keyframes/keyframes_k5.h5 --num_workers 16
"""
from __future__ import annotations

import argparse
import ast
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path("/projects/adsum-mllm-om")
EVAL_DIR = PROJECT_ROOT / "mllm" / "evaluations"
HERE = Path(__file__).resolve().parent
QP = HERE.parent  # qwen_pipeline
for p in (str(EVAL_DIR), str(QP)):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_message import extract_kframes_per_shot, FFMPEG_PATH  # noqa: E402
from frame_cache import _cached_uris, _durations, _as_list, DEFAULT_CACHE  # noqa: E402
from qwen_vl_utils import smart_resize  # noqa: E402

MAX_PIXELS = 360 * 420          # eval budget (151200)
IMAGE_FACTOR = 28               # Qwen patch alignment


def _uri_to_path(u: str) -> str:
    return u[7:] if u.startswith("file://") else u


def _resize_uint8(path: str) -> np.ndarray:
    """Load a JPEG, smart_resize to the eval budget, return RGB uint8 (H,W,3)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    h2, w2 = smart_resize(h, w, factor=IMAGE_FACTOR, max_pixels=MAX_PIXELS)
    if (w2, h2) != (w, h):
        img = img.resize((w2, h2), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def _frame_paths(info: Dict[str, Any], fps: int, jpeg_cache: Path,
                 tmp_root: str) -> Tuple[List[List[str]], List[float], Optional[str]]:
    """Frame file paths per shot. Reuse data/frames JPEGs on a cache hit;
    otherwise extract fresh into a node-local tmp dir (returned for cleanup)."""
    focal_ad_id = info.get("focal_ad_id") or Path(info["focal_relative_path"]).stem
    shotsplit = _as_list(info["focal_shotsplit"])
    durs = _durations(shotsplit, info["focal_ad_length"])
    n_shots = len(durs)

    hit = _cached_uris(jpeg_cache, focal_ad_id, fps, n_shots)
    if hit is not None:
        return [[_uri_to_path(u) for u in shot] for shot in hit], durs, None

    tmpdir = tempfile.mkdtemp(prefix=f"kf_{focal_ad_id}_", dir=tmp_root)
    video_path = (PROJECT_ROOT / info["focal_relative_path"]).resolve()
    by_shot, durs = extract_kframes_per_shot(video_path, info, output_dir=Path(tmpdir), k=fps)
    return [[_uri_to_path(u) for u in shot] for shot in by_shot], durs, tmpdir


def _build_one(args) -> Dict[str, Any]:
    """Worker: produce one ad's (n_shots,k,H,W,3) uint8 array + durations."""
    info, fps, jpeg_cache, tmp_root = args
    focal_ad_id = str(info.get("focal_ad_id") or Path(info["focal_relative_path"]).stem)
    os.environ["PATH"] = os.path.dirname(FFMPEG_PATH) + ":" + os.environ.get("PATH", "")
    tmpdir = None
    try:
        by_shot, durs, tmpdir = _frame_paths(info, fps, jpeg_cache, tmp_root)
        shots = [np.stack([_resize_uint8(p) for p in shot], axis=0) for shot in by_shot]
        # frames of one video share H,W -> dense (n_shots, k, H, W, 3)
        arr = np.stack(shots, axis=0)
        return {"ad_id": focal_ad_id, "arr": arr,
                "durs": np.asarray(durs, dtype=np.float32),
                "relative_path": info["focal_relative_path"], "error": None}
    except Exception as e:
        return {"ad_id": focal_ad_id, "arr": None, "error": f"{type(e).__name__}: {e}",
                "relative_path": info.get("focal_relative_path", "")}
    finally:
        if tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--out", required=True, help="output .h5 path")
    ap.add_argument("--jpeg_cache", default=str(DEFAULT_CACHE),
                    help="existing data/frames JPEG cache to reuse")
    ap.add_argument("--tmp_root", default=os.environ.get("TMPDIR", "/tmp"),
                    help="node-local dir for fresh extraction (NOT nfs)")
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compression", default="lzf", choices=["none", "lzf", "gzip"])
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["_ad"] = df.apply(
        lambda r: str(r.get("focal_ad_id") or Path(r["focal_relative_path"]).stem), axis=1)
    ads = df.drop_duplicates("_ad")            # frames depend only on the focal video
    if args.limit is not None:
        ads = ads.iloc[:args.limit]
    print(f"[kf_build] {len(ads)} unique ads from {len(df)} rows (fps={args.fps})")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    comp = None if args.compression == "none" else args.compression
    jpeg_cache = Path(args.jpeg_cache)

    tasks = [(r.to_dict(), args.fps, jpeg_cache, args.tmp_root)
             for _, r in ads.iterrows()]

    t0 = time.time(); ok = fail = skip = 0
    failures = []
    with h5py.File(out, "a") as h5:           # 'a' = resumable
        h5.attrs["fps"] = args.fps
        h5.attrs["max_pixels"] = MAX_PIXELS
        tasks = [t for t in tasks if t[0]["_ad"] if str(t[0].get("focal_ad_id")
                 or Path(t[0]["focal_relative_path"]).stem) not in h5]
        skip = len(ads) - len(tasks)
        print(f"[kf_build] {skip} ads already in {out.name}; building {len(tasks)}")

        ctx = mp.get_context("spawn")
        with ctx.Pool(args.num_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_build_one, tasks), 1):
                if res["error"] is not None:
                    fail += 1; failures.append((res["ad_id"], res["error"]))
                else:
                    ds = h5.create_dataset(res["ad_id"], data=res["arr"],
                                           compression=comp)
                    ds.attrs["shot_durations"] = res["durs"]
                    ds.attrs["relative_path"] = res["relative_path"]
                    ok += 1
                if i % 25 == 0 or i == len(tasks):
                    el = time.time() - t0
                    print(f"  {i}/{len(tasks)}  ok={ok} fail={fail}  "
                          f"{el:.0f}s  {i/el:.2f} ad/s", flush=True)

    print(f"\n[kf_build] done: ok={ok} fail={fail} skip={skip}  "
          f"elapsed={time.time()-t0:.0f}s  -> {out}")
    if failures:
        man = out.with_suffix(".failures.txt")
        man.write_text("\n".join(f"{a}\t{e}" for a, e in failures))
        print(f"[kf_build] {len(failures)} failures logged -> {man}")
        for a, e in failures[:5]:
            print("   ", a, e[:120])


if __name__ == "__main__":
    main()
