#!/usr/bin/env python3
"""
kf_batch_infer.py — batched Qwen3-VL eval reading keyframes from the HDF5
container (kf_build.py) via kf_dataset.H5ShotSelectionDataset.

Identical to qwen_pipeline/batch_infer.py except the data source: no ffmpeg, no
NFS small-file reads at inference time — just dense array reads from one .h5.
Reuses infer_batch + the evaluate2 result schema, so compute_tables.py and the
file-based results stay comparable (images are byte-identical; see kf equivalence
test). GPU loop never touches the filesystem hot path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path("/projects/adsum-mllm-om")
EVAL_DIR = PROJECT_ROOT / "mllm" / "evaluations"
QP = Path(__file__).resolve().parent.parent
for p in (str(EVAL_DIR), str(QP)):
    if p not in sys.path:
        sys.path.insert(0, p)

from qwen3vl_infer_video_input import load_qwen3vl  # noqa: E402
from utils import create_one_hot, calculate_metrics  # noqa: E402
from evaluate2 import load_existing_results, save_results, extract_metrics_to_csv  # noqa: E402
from batch_infer import infer_batch  # noqa: E402  (reuse the generate path verbatim)

from kf_dataset import (  # noqa: E402
    H5ShotSelectionDataset, collate_fn, frame_counts_for_csv, DynamicBatchSampler,
)


def _infer_one_by_one(model, processor, msgs, durs, gen):
    """Fallback when a batch throws: run each sample alone so a single oversized
    ad doesn't take down its whole batch. Returns a list aligned to ``msgs``;
    each item is the infer_batch result tuple, or the Exception for that sample."""
    outs = []
    for m, d in zip(msgs, durs):
        try:
            outs.append(infer_batch(model, processor, [m], [d], **gen)[0])
        except Exception as e:  # noqa: BLE001 — record per-sample, keep going
            torch.cuda.empty_cache()
            outs.append(e)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--h5", required=True, help="keyframes_k<fps>.h5 from kf_build.py")
    ap.add_argument("--output_json", default="results/kf_qwen3vl.json")
    ap.add_argument("--output_csv", default="results/kf_qwen3vl.csv")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--attn_impl", default="flash_attention_2",
                    choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Max samples per batch (cap for dynamic batching).")
    ap.add_argument("--max_frames_per_batch", type=int, default=480,
                    help="Image budget per batch for dynamic batching (n_shots*k summed).")
    ap.add_argument("--no_dynamic", action="store_true",
                    help="Disable token-budget batching; use a fixed batch_size loader.")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    output_json = Path(args.output_json); output_csv = Path(args.output_csv)
    results_dict = load_existing_results(output_json)
    print(f"Loaded {len(results_dict)} existing results from {output_json}")

    model, processor = load_qwen3vl(args.model, attn_impl=args.attn_impl)
    processor.tokenizer.padding_side = "left"

    ds = H5ShotSelectionDataset(args.csv, args.h5, fps=args.fps)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    if args.limit is not None:
        ds = Subset(ds, list(range(n)))

    if args.no_dynamic:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        persistent_workers=(args.num_workers > 0))
    else:
        counts = frame_counts_for_csv(args.csv, args.h5)[:n]
        sampler = DynamicBatchSampler(counts, args.max_frames_per_batch, args.batch_size)
        print(f"Dynamic batching: {len(sampler)} batches over {n} samples "
              f"(<= {args.batch_size} items / {args.max_frames_per_batch} frames each)")
        dl = DataLoader(ds, batch_sampler=sampler,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        persistent_workers=(args.num_workers > 0))

    for batch in tqdm(dl, desc="Batches"):
        rows = batch["row"]
        todo = [i for i, r in enumerate(rows) if r not in results_dict]
        if not todo:
            continue
        bad = [i for i in todo if batch["messages"][i] is None]
        for i in bad:
            results_dict[rows[i]] = {"row": rows[i], "video": batch["video_path"][i],
                                     "error": batch["error"][i]}
        todo = [i for i in todo if batch["messages"][i] is not None]
        if not todo:
            save_results(results_dict, output_json)
            continue

        msgs = [batch["messages"][i] for i in todo]
        durs = [batch["shot_durations"][i] for i in todo]
        gen = dict(max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_p=args.top_p)
        try:
            outs = infer_batch(model, processor, msgs, durs, **gen)
        except Exception:
            # A batch-level failure (usually CUDA OOM) shouldn't poison every row:
            # free memory and retry one sample at a time so only the genuine
            # offender is recorded as an error.
            torch.cuda.empty_cache()
            outs = _infer_one_by_one(model, processor, msgs, durs, gen)

        for k, i in enumerate(todo):
            res = outs[k]
            if isinstance(res, Exception):
                results_dict[rows[i]] = {"row": rows[i], "video": batch["video_path"][i],
                                         "error": str(res)}
                continue
            out, ok, total, selected_pred = res
            one_hot_gt = batch["label_one_hot"][i]
            one_hot_pred = create_one_hot(batch["all_shots"][i], selected_pred)
            metrics = calculate_metrics(one_hot_gt, one_hot_pred)
            results_dict[rows[i]] = {
                "row": rows[i], "video": batch["video_path"][i], "answers": out,
                "ok": ok, "total": total, "selected": selected_pred,
                "gt": batch["selected_gt"][i],
                "accuracy": metrics["accuracy"], "precision": metrics["precision"],
                "recall": metrics["recall"], "f1_score": metrics["f1_score"],
                "iou": metrics["iou"], "auc": metrics["auc"],
                "unmatched_shots": batch["unmatched_shots"][i],
            }
        save_results(results_dict, output_json)

    print(f"\nAll results saved to {output_json}")
    extract_metrics_to_csv(output_json, output_csv)
    print(f"Metrics saved to {output_csv}")


if __name__ == "__main__":
    main()
