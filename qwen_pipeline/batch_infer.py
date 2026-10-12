#!/usr/bin/env python3
"""
batch_infer.py — batched Qwen3-VL evaluation for the shot-selection task.

Generalizes evaluations/qwen3vl_infer_video_input.infer_one to N samples per
model.generate call (configurable --batch_size), driven by the DataLoader in
dataset.py (parallel ffmpeg + on-disk frame cache). Emits the SAME JSON/CSV
result schema as evaluations/evaluate2.py, so compute_tables.py keeps working.

Greedy decoding (temperature=0) is deterministic, so batched results must equal
single-sample results — verify with --batch_size 1 vs 2 on the same --limit.

Env: requires transformers>=4.57 (Qwen3VLForConditionalGeneration):
  /projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python batch_infer.py ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Subset
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

PROJECT_ROOT = Path("/projects/adsum-mllm-om")
EVAL_DIR = PROJECT_ROOT / "mllm" / "evaluations"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from qwen3vl_infer_video_input import (  # noqa: E402
    load_qwen3vl, parse_selected_indices, validate_selection,
)
from utils import create_one_hot, calculate_metrics  # noqa: E402
from evaluate2 import load_existing_results, save_results, extract_metrics_to_csv  # noqa: E402

from dataset import build_dataloader, ShotSelectionDataset, collate_fn  # noqa: E402
from torch.utils.data import DataLoader


@torch.inference_mode()
def infer_batch(model, processor, messages_list: List[List[Dict[str, Any]]],
                shot_durations_list: List[List[float]],
                max_new_tokens: int, temperature: float, top_p: float):
    """Run N samples through one generate call. Returns parallel lists
    (out, ok, total, selected) — mirrors infer_one per sample."""
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages_list]
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages_list, return_video_kwargs=True)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **(video_kwargs or {}),
    ).to(model.device)

    do_sample = temperature > 0.0
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
    )
    # Left padding => every input row has the same length; slice off the prompt.
    in_len = inputs.input_ids.shape[1]
    trimmed = [out[in_len:] for out in gen_ids]
    out_texts = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    results = []
    for out, durs in zip(out_texts, shot_durations_list):
        out = out.strip()
        selected = parse_selected_indices(out)
        ok, total, selected = validate_selection(selected, durs)
        results.append((out, ok, total, selected))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output_json", default="results/qwen3vl_batch.json")
    ap.add_argument("--output_csv", default="results/qwen3vl_batch.csv")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--max_pixels", type=int, default=360 * 420)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Only first N rows (verification).")
    args = ap.parse_args()

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    results_dict = load_existing_results(output_json)
    print(f"Loaded {len(results_dict)} existing results from {output_json}")

    model, processor = load_qwen3vl(args.model, attn_impl=args.attn_impl)
    # Decoder generation needs left padding for correct batched output.
    processor.tokenizer.padding_side = "left"

    ds = ShotSelectionDataset(args.csv, fps=args.fps, max_pixels=args.max_pixels)
    if args.limit is not None:
        ds = Subset(ds, list(range(min(args.limit, len(ds)))))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_fn,
                    persistent_workers=(args.num_workers > 0))

    for batch in tqdm(dl, desc="Batches"):
        rows = batch["row"]
        # Skip a batch only if every row is already done.
        todo = [i for i, r in enumerate(rows) if r not in results_dict]
        if not todo:
            continue

        # Rows whose frame extraction failed (corrupt video etc.) arrive with
        # messages=None; record them as errors and never feed them to generate.
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
        try:
            outs = infer_batch(model, processor, msgs, durs,
                               max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_p=args.top_p)
        except Exception as e:  # whole-batch failure -> record per-row error
            for i in todo:
                results_dict[rows[i]] = {"row": rows[i], "video": batch["video_path"][i],
                                         "error": str(e)}
            save_results(results_dict, output_json)
            continue

        for k, i in enumerate(todo):
            out, ok, total, selected_pred = outs[k]
            all_shots = batch["all_shots"][i]
            one_hot_gt = batch["label_one_hot"][i]
            one_hot_pred = create_one_hot(all_shots, selected_pred)
            metrics = calculate_metrics(one_hot_gt, one_hot_pred)
            results_dict[rows[i]] = {
                "row": rows[i],
                "video": batch["video_path"][i],
                "answers": out,
                "ok": ok,
                "total": total,
                "selected": selected_pred,
                "gt": batch["selected_gt"][i],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "iou": metrics["iou"],
                "auc": metrics["auc"],
                "unmatched_shots": batch["unmatched_shots"][i],
            }
        save_results(results_dict, output_json)

    print(f"\nAll results saved to {output_json}")
    extract_metrics_to_csv(output_json, output_csv)
    print(f"Metrics saved to {output_csv}")


if __name__ == "__main__":
    main()
