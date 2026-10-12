#!/usr/bin/env python3
"""
kf_dataset.py — Dataset that reads keyframes from the HDF5 container (kf_build.py)
instead of per-shot JPEG files. Same task/label schema as qwen_pipeline/dataset.py,
same prompt (build_message.build_system_prompt), but images are passed as in-memory
PIL objects decoded from the HDF5 arrays — no ffmpeg, no NFS small-file reads.

HDF5 handles can't be shared across forked workers, so the file is opened lazily
per process inside __getitem__ (never in __init__).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import h5py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path("/projects/adsum-mllm-om")
EVAL_DIR = PROJECT_ROOT / "mllm" / "evaluations"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from build_message import build_system_prompt  # noqa: E402
from utils import extract_shots_from_mapping, create_one_hot  # noqa: E402


def ad_id_of(info: Dict[str, Any]) -> str:
    """Resolve the HDF5 key for a CSV row: focal_ad_id, else the video stem."""
    return str(info.get("focal_ad_id") or Path(info["focal_relative_path"]).stem)


def assemble_messages_pil(frames_by_shot: List[List[Image.Image]],
                          shot_durations: List[float], fps: int):
    """Mirror build_message.assemble_messages exactly, but with PIL images
    (already resized to the eval budget) instead of file:// URIs."""
    n_shots = len(shot_durations)
    system_prompt = build_system_prompt(fps)
    user_prompt = (
        f"This advertisement contains {n_shots} shots. The duration of each shot (in seconds) is: "
        + json.dumps({f"shot{i}": round(d, 2) for i, d in enumerate(shot_durations, start=1)})
        + "\n" + "The frames of each shot are below: \n"
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for i, frames in enumerate(frames_by_shot, start=1):
        content.append({"type": "text", "text": f"[Shot {i}]"})
        for img in frames:
            content.append({"type": "image", "image": img})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


class H5ShotSelectionDataset(Dataset):
    def __init__(self, csv_path, h5_path, fps: int = 5):
        self.df = pd.read_csv(csv_path)
        self.h5_path = str(h5_path)
        self.fps = fps
        self._h5 = None   # opened lazily per worker

    def __len__(self):
        return len(self.df)

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _ad_id(self, info):
        return ad_id_of(info)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.df.loc[idx].to_dict()
        ad_id = self._ad_id(info)
        try:
            ds = self._file()[ad_id]
            arr = ds[:]                                   # (n_shots, k, H, W, 3)
            durs = [float(x) for x in ds.attrs["shot_durations"]]
            frames_by_shot = [[Image.fromarray(arr[s, j]) for j in range(arr.shape[1])]
                              for s in range(arr.shape[0])]
            messages = assemble_messages_pil(frames_by_shot, durs, self.fps)
            selected_gt, all_shots = extract_shots_from_mapping(info["mapping"])
            label_one_hot = create_one_hot(all_shots, selected_gt)
            error = None
        except Exception as e:
            messages, durs, all_shots, selected_gt, label_one_hot = None, [], [], [], []
            error = f"{type(e).__name__}: {e}"

        return {
            "row": idx,
            "video_path": info["focal_relative_path"],
            "messages": messages,
            "shot_durations": durs,
            "all_shots": all_shots,
            "selected_gt": selected_gt,
            "label_one_hot": label_one_hot,
            "unmatched_shots": int(info.get("unmatched_shots", 0)),
            "error": error,
        }


def collate_fn(batch):
    return {k: [b[k] for b in batch] for k in batch[0].keys()}


def frame_counts_for_csv(csv_path, h5_path) -> List[int]:
    """Per-row image count (n_shots * k) read from HDF5 shape metadata only — no
    array decode. Used to size dynamic batches. Missing ids (ad not in the .h5)
    get count 1 so they are still scheduled; they then error in __getitem__."""
    df = pd.read_csv(csv_path)
    counts: List[int] = []
    with h5py.File(str(h5_path), "r") as f:
        for idx in range(len(df)):
            ad_id = ad_id_of(df.loc[idx].to_dict())
            try:
                s = f[ad_id].shape            # (n_shots, k, H, W, 3)
                counts.append(int(s[0]) * int(s[1]))
            except Exception:
                counts.append(1)
    return counts


class DynamicBatchSampler:
    """Greedy, order-preserving batch sampler that bounds total images per batch.

    Packs consecutive dataset indices until adding the next would exceed
    ``max_frames`` total images, or the batch already holds ``max_bs`` items.
    A single sample larger than ``max_frames`` is emitted alone (never dropped).
    Order is preserved so the run stays resumable (results keyed by CSV row).
    """

    def __init__(self, frame_counts: List[int], max_frames: int, max_bs: int):
        self.fc = list(frame_counts)
        self.max_frames = max_frames
        self.max_bs = max_bs

    def __iter__(self):
        batch: List[int] = []
        tot = 0
        for i, c in enumerate(self.fc):
            if batch and (len(batch) >= self.max_bs or tot + c > self.max_frames):
                yield batch
                batch, tot = [], 0
            batch.append(i)
            tot += c
        if batch:
            yield batch

    def __len__(self):
        n = 0
        cur = 0
        tot = 0
        for c in self.fc:
            if cur and (cur >= self.max_bs or tot + c > self.max_frames):
                n += 1
                cur, tot = 0, 0
            cur += 1
            tot += c
        return n + (1 if cur else 0)
