"""
Fused keyframe-extraction + embedding stage (s1b + s1c in one pass).

Why this exists
---------------
The decoupled stages run sequentially: s1b decodes every video and writes ~5
tiny JPEGs per shot to disk (a metadata storm on NFS), THEN s1c re-reads/re-decodes
those JPEGs and runs Swin3D. This module fuses them into one CPU->GPU pipeline:

    N CPU decode workers  --(bounded queue)-->  1 GPU embed worker  --> parent writes H5

The video decode (CPU) of one video overlaps the GPU forward of earlier videos, so
keyframe extraction effectively hides behind the embedding pass. No keyframe files
are written -- the only persisted output is the per-brand embeddings .h5, identical
in schema to s1_embedding.py.

Byte-identity
-------------
Embeddings must match the reference (cosine >= 0.9999 gate). The reference pixels
went through a JPEG round-trip (PIL save q95 subsampling=0 -> PIL open). We reproduce
that EXACTLY with an in-memory JPEG encode+decode (same params) before the Swin3D
transform, so the pixels -- and therefore the embeddings -- are bit-for-bit the same
as the on-disk path. The JPEG bytes are simply never written to disk.

Mirrors the two-stage pattern in s1_shot_detection.run_parallel (bounded queue =
backpressure; all H5 writes happen in the parent, so there is no cross-process race).
"""

import os

# Cap math-library thread pools to 1 BEFORE numpy/torch/cv2 import and read them.
# 24 spawn workers each defaulting their BLAS/torch/cv2 pools to the full core count
# => 24 x N threads fighting over N cores (load avg >> cores). One thread per worker
# lets the 24 workers actually run in parallel and fill the GPU queue.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import io
import logging
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

import config
import utils
from s1_keyframes import KeyframeExtractor

logger = logging.getLogger(__name__)

# Swin3D input normalization (must match s1_embedding.Swin3DEmbedder.custom_transform).
_RESIZE = 256
_CROP = 224
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Pure helpers (shared by parent for resume-planning and by CPU workers)
# ---------------------------------------------------------------------------
def plan_valid_shots(shot_splits, ad_id, percentiles):
    """Return [(shot_id, [5 frame indices])] for the ad's VALID shots.

    Reproduces KeyframeExtractor.extract_keyframes' planning + find_valid_shots'
    filter: only "long" shots (>= 5 frames) that yield all 5 percentile frames
    inside the clip are valid (short shots are dropped, exactly as the embedding
    stage drops them). shot_id matches `{ad_id}_shot_NN`.
    """
    ss = list(shot_splits)
    if not ss:
        return []
    if ss[0] != 0:
        ss = [0] + ss
    total_frames = ss[-1]

    out = []
    for i in range(len(ss) - 1):
        start = ss[i]
        end = min(ss[i + 1], total_frames)
        shot_len = end - start
        if shot_len < 5:                       # short shot -> not a valid 5-frame shot
            continue
        idxs = []
        for p in percentiles:
            ti = start + int(p * shot_len)
            if ti >= total_frames:
                break
            idxs.append(ti)
        if len(idxs) == len(percentiles):       # all 5 percentile frames in range
            out.append((f"{ad_id}_shot_{i + 1:02d}", idxs))
    return out


def jpeg_roundtrip(arr, quality, subsampling):
    """uint8 (H,W,3) -> identical bytes to a written JPEG -> decoded back to array.

    Done fully in memory; reproduces the exact pixels the on-disk path produced.
    """
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return np.array(Image.open(buf))


def transform_clip(frames):
    """List of 5 uint8 (H,W,3) frames -> (C, T, H, W) float tensor for Swin3D.

    Identical ops to s1_embedding.Swin3DEmbedder.custom_transform (resize 256
    bilinear -> center-crop 224 -> ToTensor -> Normalize), minus the leading batch
    dim. Runs on CPU in the decode workers so the single GPU worker stays light.
    """
    resize = T.Resize(_RESIZE, interpolation=T.InterpolationMode.BILINEAR)
    crop = T.CenterCrop(_CROP)
    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=_MEAN, std=_STD)

    out = []
    for arr in frames:
        img = Image.fromarray(arr.astype(np.uint8))
        img = resize(img)
        img = crop(img)
        t = normalize(to_tensor(img))          # (C, H, W)
        out.append(t)
    clip = torch.stack(out)                    # (T, C, H, W)
    return clip.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)


# ---------------------------------------------------------------------------
# Stage 1: CPU decode worker
# ---------------------------------------------------------------------------
def _decode_worker(task_q, clip_q, percentiles, quality, subsampling, done_shots):
    """Decode one video's valid keyframes and push per-shot transformed clips.

    Blocks on a full clip_q (backpressure -> bounded RAM). Emits one message per
    video: {meta, shots: [(shot_id, clip_float32(C,T,H,W))]}; on failure emits an
    error message so the video is logged, not silently dropped.
    """
    import cv2
    cv2.setNumThreads(0)          # cv2 decode/resize single-threaded per worker
    torch.set_num_threads(1)      # torchvision ToTensor/Normalize are torch ops
    extractor = KeyframeExtractor()
    while True:
        item = task_q.get()
        if item is None:
            clip_q.put(None)                   # one STOP token per decode worker
            break
        try:
            plan = plan_valid_shots(item["shot_splits"], item["ad_id"], percentiles)
            plan = [(sid, idxs) for sid, idxs in plan if sid not in done_shots]
            if not plan:
                clip_q.put({"meta": item, "shots": []})
                continue

            wanted = sorted({i for _, idxs in plan for i in idxs})
            frames_by_idx = extractor._decode_indices(item["video_path"], wanted)

            shots = []
            for shot_id, idxs in plan:
                frames = [jpeg_roundtrip(frames_by_idx[i], quality, subsampling)
                          for i in idxs]
                clip = transform_clip(frames).numpy().astype("float32")  # (C,T,H,W)
                shots.append((shot_id, clip))
            clip_q.put({"meta": item, "shots": shots})
        except Exception as e:
            clip_q.put({"meta": item, "error": str(e), "error_type": type(e).__name__})


# ---------------------------------------------------------------------------
# Stage 2: GPU embed worker
# ---------------------------------------------------------------------------
def _gpu_worker(clip_q, result_q, batch_size, num_decode_workers):
    """Load Swin3D ONCE, batch incoming clips (across videos) and forward them.

    Pushes results as {brand_id, brand_name, shot_ids[], embeddings(np B,1024)}.
    Stops after it has seen one STOP token from every decode worker.
    """
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from s1_embedding import Swin3DEmbedder
    embedder = Swin3DEmbedder()

    pending = []   # list of (brand_id, brand_name, shot_id, clip_np)

    def flush(force=False):
        while pending and (force or len(pending) >= batch_size):
            chunk = pending[:batch_size]
            del pending[:batch_size]
            batch = torch.from_numpy(np.stack([c for *_, c in chunk])).to(embedder.device)
            embs = embedder._forward_batch(batch).cpu().numpy()
            by_brand = defaultdict(lambda: ([], [], None))
            for (bid, bname, sid, _c), emb in zip(chunk, embs):
                sids, es, _ = by_brand[bid]
                sids.append(sid)
                es.append(emb)
                by_brand[bid] = (sids, es, bname)
            for bid, (sids, es, bname) in by_brand.items():
                result_q.put({"brand_id": bid, "brand_name": bname,
                              "shot_ids": sids, "embeddings": np.asarray(es, dtype="float32")})

    stops = 0
    while stops < num_decode_workers:
        msg = clip_q.get()
        if msg is None:
            stops += 1
            continue
        meta = msg["meta"]
        if "error" in msg:
            result_q.put({"meta": meta, "error": msg["error"],
                          "error_type": msg.get("error_type")})
            continue
        for shot_id, clip in msg["shots"]:
            pending.append((meta["brand_id"], meta["brand_name"], shot_id, clip))
        flush(force=False)
        result_q.put({"video_done": meta["ad_id"]})
    flush(force=True)
    result_q.put({"gpu_done": True})


# ---------------------------------------------------------------------------
# H5 writer (same schema as s1_embedding.EmbeddingProcessor._write_brand_h5)
# ---------------------------------------------------------------------------
def _write_brand_h5(embeddings_dir, brand_id, brand_name, shot_ids, embeddings, batch_size):
    h5_path = embeddings_dir / f"{brand_id}_{brand_name}_embeddings.h5"
    embeddings = np.asarray(embeddings, dtype="float32")
    mode = "a" if h5_path.exists() else "w"
    with h5py.File(h5_path, mode) as hf:
        if "embeddings" not in hf:
            hf.create_dataset("embeddings", (0, 1024), maxshape=(None, 1024),
                              dtype="float32", chunks=(batch_size, 1024), compression="gzip")
            hf.create_dataset("brand_ids", (0,), maxshape=(None,), dtype="int32",
                              chunks=(batch_size,), compression="gzip")
            hf.create_dataset("shot_ids", (0,), maxshape=(None,), dtype="S100",
                              chunks=(batch_size,), compression="gzip")
        cur = hf["embeddings"].shape[0]
        new = cur + len(shot_ids)
        hf["embeddings"].resize(new, axis=0)
        hf["brand_ids"].resize(new, axis=0)
        hf["shot_ids"].resize(new, axis=0)
        hf["embeddings"][cur:new] = embeddings
        hf["brand_ids"][cur:new] = np.full(len(shot_ids), brand_id, dtype="int32")
        hf["shot_ids"][cur:new] = np.array([s.encode("ascii") for s in shot_ids])
        hf.flush()
    return h5_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class FusedKeyframeEmbedProcessor:
    """Decode keyframes and embed them in one CPU->GPU pipeline (no keyframe files)."""

    def __init__(self, data_df, shotsplits_dir=None, video_base_dir=None,
                 output_dir=None, batch_size=None, percentiles=None,
                 quality=None, subsampling=None):
        self.data_df = data_df
        self.shotsplits_dir = Path(shotsplits_dir or config.SHOTSPLITS_DIR)
        self.video_base_dir = Path(video_base_dir or config.VIDEO_BASE_DIR)
        self.embeddings_dir = Path(output_dir or config.EMBEDDINGS_DIR)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        self.percentiles = percentiles or config.KEYFRAME_PERCENTILES
        self.quality = quality if quality is not None else config.IMAGE_QUALITY
        self.subsampling = subsampling if subsampling is not None else config.IMAGE_SUBSAMPLING

    # --- shotsplits lookup (mirrors KeyframeProcessor._load_shotsplits) ---
    def _load_shotsplits(self):
        mapping = {}
        for f in self.shotsplits_dir.glob("*_shotsplits.json"):
            try:
                df = pd.read_json(f, lines=True)
            except ValueError:
                continue
            if "shotsplit" not in df.columns:
                continue
            for _, row in df.drop_duplicates(subset="ad_id", keep="first").iterrows():
                mapping[str(row["ad_id"])] = utils.parse_shot_splits(row["shotsplit"])
        return mapping

    def _resolve_video_path(self, row):
        rel = row.get("relative_path")
        if isinstance(rel, str) and rel:
            return self.video_base_dir.parent / rel
        return self.video_base_dir / row["brand_name"] / f"{row['ad_id']}.mp4"

    def _existing_shots(self):
        """All shot_ids already embedded across this batch's brand H5s (resume)."""
        done = set()
        brands = {(r["brand_id"], r["brand_name"]) for _, r in self.data_df.iterrows()}
        for brand_id, brand_name in brands:
            h5_path = self.embeddings_dir / f"{brand_id}_{brand_name}_embeddings.h5"
            if not h5_path.exists():
                continue
            try:
                with h5py.File(h5_path, "r") as hf:
                    if "shot_ids" in hf:
                        done.update(s.decode("ascii") for s in hf["shot_ids"][:])
            except Exception:
                pass
        return done

    def _build_tasks(self, shot_map, done_shots, skip_existing):
        """Per-video work items, skipping videos whose valid shots are all done."""
        tasks = []
        missing_shots = missing_video = skipped = 0
        for _, row in self.data_df.iterrows():
            ad_id = str(row["ad_id"])
            shots = shot_map.get(ad_id)
            if shots is None:
                missing_shots += 1
                continue
            video_path = self._resolve_video_path(row)
            if not video_path.exists():
                missing_video += 1
                logger.warning(f"Missing video file: {video_path}")
                continue
            if skip_existing:
                planned = [sid for sid, _ in plan_valid_shots(shots, ad_id, self.percentiles)]
                if planned and all(sid in done_shots for sid in planned):
                    skipped += 1
                    continue
            tasks.append({
                "ad_id": ad_id,
                "brand_id": int(row["brand_id"]),
                "brand_name": row["brand_name"],
                "video_path": str(video_path),
                "shot_splits": list(shots),
            })
        logger.info(
            f"Batch: {len(self.data_df)} videos | {skipped} already embedded | "
            f"{missing_shots} without shotsplits | {missing_video} missing files | "
            f"{len(tasks)} to process"
        )
        return tasks

    def run(self, num_decode_workers=24, queue_size=None, skip_existing=True):
        import multiprocessing as mp
        import threading

        shot_map = self._load_shotsplits()
        done_shots = self._existing_shots() if skip_existing else set()
        tasks = self._build_tasks(shot_map, done_shots, skip_existing)
        if not tasks:
            logger.info("Nothing to process.")
            return 0

        ctx = mp.get_context("spawn")
        queue_size = queue_size or max(2, num_decode_workers * 2)
        task_q = ctx.Queue()
        clip_q = ctx.Queue(maxsize=queue_size)     # bounded -> backpressure
        result_q = ctx.Queue()

        decode_procs = [
            ctx.Process(target=_decode_worker,
                        args=(task_q, clip_q, list(self.percentiles),
                              self.quality, self.subsampling, done_shots),
                        daemon=True)
            for _ in range(num_decode_workers)
        ]
        gpu_proc = ctx.Process(target=_gpu_worker,
                               args=(clip_q, result_q, self.batch_size, num_decode_workers),
                               daemon=True)
        for p in decode_procs:
            p.start()
        gpu_proc.start()

        for t in tasks:
            task_q.put(t)
        for _ in range(num_decode_workers):
            task_q.put(None)

        logger.info(f"Pipeline: {num_decode_workers} decode workers -> "
                    f"queue(maxsize={queue_size}) -> 1 GPU worker (batch={self.batch_size})")

        # Accumulate per-brand, write H5 at the end (one append per brand).
        results = defaultdict(lambda: ([], [], None))   # brand_id -> (shot_ids, embs, name)
        failed = 0
        pbar = tqdm(total=len(tasks), desc="Fused keyframe+embed")
        gpu_done = False
        while not gpu_done:
            res = result_q.get()
            if res.get("gpu_done"):
                gpu_done = True
            elif res.get("video_done"):
                pbar.update(1)
            elif "error" in res:
                logger.error(f"Error on {res['meta']['ad_id']}: {res['error']}")
                failed += 1
                pbar.update(1)
            else:
                bid = res["brand_id"]
                sids, es, _ = results[bid]
                sids.extend(res["shot_ids"])
                es.extend(res["embeddings"])
                results[bid] = (sids, es, res["brand_name"])
        pbar.close()
        for p in decode_procs:
            p.join()
        gpu_proc.join()

        total = 0
        for bid, (sids, es, bname) in results.items():
            if not sids:
                continue
            path = _write_brand_h5(self.embeddings_dir, bid, bname, sids, es, self.batch_size)
            total += len(sids)
            logger.info(f"  brand {bid} ({bname}): {len(sids)} embeddings -> {path.name}")

        logger.info(f"Fused stage done: {total} embeddings, {failed} video errors "
                    f"-> {self.embeddings_dir}")
        return total


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fused keyframe-extraction + Swin3D embedding (s1b+s1c, no keyframe files)")
    parser.add_argument("--data_csv", required=True, help="Batch videos CSV")
    parser.add_argument("--shotsplits_dir", default=config.SHOTSPLITS_DIR,
                        help="Dir with per-brand *_shotsplits.json from the shot stage")
    parser.add_argument("--video_dir", default=config.VIDEO_BASE_DIR)
    parser.add_argument("--output_dir", default=config.EMBEDDINGS_DIR,
                        help="Where to write per-brand *_embeddings.h5")
    parser.add_argument("--num_decode_workers", type=int, default=24,
                        help="CPU decode/transform workers feeding the GPU")
    parser.add_argument("--batch_size", type=int, default=config.EMBEDDING_BATCH_SIZE,
                        help="GPU forward batch (shots per forward)")
    parser.add_argument("--queue_size", type=int, default=None,
                        help="Bounded clip-queue size (default: 2x decode workers)")
    parser.add_argument("--no_skip_existing", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    data_df = pd.read_csv(args.data_csv)
    logger.info(f"Loaded {len(data_df)} videos from "
                f"{data_df['brand_name'].nunique()} brands ({args.data_csv})")

    processor = FusedKeyframeEmbedProcessor(
        data_df=data_df,
        shotsplits_dir=args.shotsplits_dir,
        video_base_dir=args.video_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    processor.run(num_decode_workers=args.num_decode_workers,
                  queue_size=args.queue_size,
                  skip_existing=not args.no_skip_existing)


if __name__ == "__main__":
    main()
