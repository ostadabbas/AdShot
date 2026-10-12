#!/usr/bin/env python3
"""
merge_adapter.py — merge a trained LoRA adapter into the base Qwen3-VL-8B weights
and save a standalone model directory, so the EXISTING eval entry point
(keyframes/kf_batch_infer.py --model <dir>) loads it with zero eval-code changes.

Endpoint (run on a GPU node; merge needs the base weights in memory):

  /projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python training/merge_adapter.py \
      --base Qwen/Qwen3-VL-8B-Instruct \
      --adapter training/runs/qwen3vl8b_lora_v1 \
      --out     training/runs/qwen3vl8b_lora_v1_merged

Inputs : --base (HF id or path), --adapter (dir saved by train_lora.py)
Output : --out  (a full model dir: merged weights + processor) loadable by load_qwen3vl()
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel


def merge(base: str, adapter: str, out: str):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()          # fold LoRA deltas into base weights
    model.save_pretrained(str(out_dir))

    # processor was saved alongside the adapter by train_lora.py; fall back to base.
    try:
        processor = AutoProcessor.from_pretrained(adapter)
    except Exception:
        processor = AutoProcessor.from_pretrained(base)
    processor.save_pretrained(str(out_dir))
    print(f"Merged model + processor written to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into base weights.")
    ap.add_argument("--base", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", required=True, help="Adapter dir from train_lora.py")
    ap.add_argument("--out", required=True, help="Output standalone model dir")
    args = ap.parse_args()
    merge(args.base, args.adapter, args.out)


if __name__ == "__main__":
    main()
