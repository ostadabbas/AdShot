#!/usr/bin/env python3
"""
dataset.py — PyTorch Dataset/DataLoader for the Qwen3-VL shot-selection task.

One item per fullmatch row:
  - input  : the focal (30s) video as k frames/shot (cached), wrapped in the
             exact eval prompt via build_message.assemble_messages.
  - label  : one-hot over focal shots (1 = shot appears in the short ad).
             Built with utils.extract_shots_from_mapping + create_one_hot.

The Dataset yields the assembled `messages` plus label metadata. Tokenization /
processor padding happens downstream (batch_infer.py or a trainer collate),
because that needs the model's processor. collate_fn here just gathers lists.

frame extraction (ffmpeg) is the per-item bottleneck, so DataLoader num_workers
parallelizes it and the on-disk frame cache makes subsequent epochs cheap.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("/projects/adsum-mllm-om")
EVAL_DIR = PROJECT_ROOT / "mllm" / "evaluations"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from build_message import assemble_messages, FFMPEG_PATH  # noqa: E402
from utils import extract_shots_from_mapping, create_one_hot  # noqa: E402

from frame_cache import get_frame_uris, DEFAULT_CACHE  # noqa: E402


def _worker_init(_):
    # ffmpeg lives in the `shot` env; make sure workers can find it on PATH too.
    os.environ["PATH"] = os.path.dirname(FFMPEG_PATH) + ":" + os.environ.get("PATH", "")


class ShotSelectionDataset(Dataset):
    def __init__(self, csv_path, fps: int = 5, max_pixels: int = 360 * 420,
                 cache_root: Path = DEFAULT_CACHE, project_root: Path = PROJECT_ROOT):
        self.df = pd.read_csv(csv_path)
        self.fps = fps
        self.max_pixels = max_pixels
        self.cache_root = Path(cache_root)
        self.project_root = Path(project_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.df.loc[idx].to_dict()
        video_path = (self.project_root / info["focal_relative_path"]).resolve()

        # A corrupt/short video can make ffmpeg seek past EOF -> no frame written
        # -> FileNotFoundError deep in extract_kframes_per_shot. In a DataLoader
        # worker that kills the whole run, so isolate it per-row (mirrors the
        # per-row try/except in evaluate2.py) and surface it as an error sentinel.
        try:
            frame_uris, shot_durations = get_frame_uris(
                video_path, info, fps=self.fps, cache_root=self.cache_root)
            messages, _, _ = assemble_messages(
                frame_uris, shot_durations, fps=self.fps, max_pixels=self.max_pixels)

            selected_gt, all_shots = extract_shots_from_mapping(info["mapping"])
            label_one_hot = create_one_hot(all_shots, selected_gt)
            error = None
        except Exception as e:
            messages = None
            shot_durations, all_shots, selected_gt, label_one_hot = [], [], [], []
            error = f"{type(e).__name__}: {e}"

        return {
            "row": idx,
            "video_path": str(video_path),
            "messages": messages,
            "shot_durations": shot_durations,   # focal shot durations (validate_selection)
            "all_shots": all_shots,
            "selected_gt": selected_gt,
            "label_one_hot": label_one_hot,
            "unmatched_shots": int(info.get("unmatched_shots", 0)),
            "error": error,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Gather per-sample fields into parallel lists; processor padding is done
    by the consumer (batch_infer), which needs the model processor."""
    return {k: [b[k] for b in batch] for k in batch[0].keys()}


def build_dataloader(csv_path, batch_size: int = 4, num_workers: int = 8,
                     shuffle: bool = False, fps: int = 5,
                     max_pixels: int = 360 * 420, **kw) -> DataLoader:
    ds = ShotSelectionDataset(csv_path, fps=fps, max_pixels=max_pixels, **kw)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate_fn, worker_init_fn=_worker_init,
        persistent_workers=(num_workers > 0),
    )


if __name__ == "__main__":
    import argparse
    HERE = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "data" / "test.csv"))
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    dl = build_dataloader(args.csv, batch_size=args.batch_size,
                          num_workers=args.num_workers, fps=args.fps)
    batch = next(iter(dl))
    print(f"batch keys: {list(batch.keys())}")
    print(f"batch size: {len(batch['messages'])}")
    for i in range(len(batch["messages"])):
        m = batch["messages"][i]
        n_img = sum(1 for c in m[1]["content"] if c.get("type") == "image")
        print(f"  sample{i}: shots={len(batch['all_shots'][i])} "
              f"label_len={len(batch['label_one_hot'][i])} "
              f"selected={batch['selected_gt'][i]} images={n_img} "
              f"msg_roles={[mm['role'] for mm in m]}")
        assert len(batch["label_one_hot"][i]) == len(batch["all_shots"][i])
    print("OK: dataset + collate")
