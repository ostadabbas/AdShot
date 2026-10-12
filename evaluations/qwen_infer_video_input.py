#!/usr/bin/env python3
"""
qwen_infer_with_eval_frames.py

- Uses FFMPEG_PATH to extract a 15s clip (no PATH dependency).
- Computes the exact ORIGINAL frame indices (0-based) used for eval:
    N = round(clip_len_sec * fps)
    t_i = start_sec + i/fps
    idx_i = floor(t_i * orig_fps), clamped to [0, total_frames-1]
- Runs Qwen2.5-VL-7B inference on that 15s clip using the same fps you pass.

Install:
  pip install "git+https://github.com/huggingface/transformers" accelerate
  pip install "qwen-vl-utils[decord]"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import json
import re
import ast
from typing import Dict, Any, List, Optional
import itertools

from build_message import build_messages

# >>> YOUR ENVIRONMENT <<<
FFMPEG_PATH = "/projects/adsum-mllm-om/miniconda3/envs/shot/bin/ffmpeg"
os.environ["PATH"] = "/projects/adsum-mllm-om/miniconda3/envs/shot/bin:" + os.environ["PATH"]

SHORT_THRESHOLD = 0.5  # seconds
LONG_THRESHOLD = 4.0


def unique_in_range(seq, lo, hi):
    out, seen = [], set()
    for x in seq:
        if lo <= x <= hi and x not in seen:
            out.append(x); seen.add(x)
    return out

def selection_total_seconds(selected_shots, shot_durations):
    return sum(shot_durations[s-1] for s in selected_shots)

def validate_selection(selected_shots, shot_durations, min_k=5, max_k=12, min_s=14.5, max_s=15.5):
    n = len(shot_durations)
    selected_shots = unique_in_range(selected_shots, 1, n)
    total = selection_total_seconds(selected_shots, shot_durations)
    ok = (min_k <= len(selected_shots) <= max_k) and (min_s <= total <= max_s) and (len(selected_shots) < n)
    return ok, total, selected_shots


def parse_selected_indices(out: str) -> List[int]:
    # 1) Try JSON parse first (preferred)
    try:
        m = re.search(r"\{.*\}", out, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in output")

        obj = json.loads(m.group(0))
        shots = obj.get("selected_shots", [])

        # ["shot4", "shot17"] -> [4, 17]
        return [
            int(mm.group())
            for s in shots
            if (mm := re.search(r"\d+", str(s))) is not None
        ]
    except Exception:
        print("Selection parsing exception seen")
        # 2) Fallback: extract *all* ints from the raw output (old behavior)
        nums = re.findall(r"-?\d+", out)
        return [int(x) for x in nums]



def extract_clip_ffmpeg(src: Path, dst: Path, start_sec: float, clip_len_sec: float, ffmpeg_bin: str) -> None:
    ffmpeg_bin_path = Path(ffmpeg_bin)
    if not ffmpeg_bin_path.exists():
        raise FileNotFoundError(f"ffmpeg not found at: {ffmpeg_bin}")

    # Accurate cut: place -ss after -i (slower but aligns exactly)
    cmd_x264 = [
        str(ffmpeg_bin_path), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-ss", str(start_sec),
        "-t", str(clip_len_sec),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    try:
        subprocess.run(cmd_x264, check=True)
        return
    except Exception:
        # Fallback (less accurate if keyframe-limited, but better than failing)
        cmd_copy = [
            str(ffmpeg_bin_path), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start_sec),
            "-t", str(clip_len_sec),
            "-i", str(src),
            "-c", "copy",
            str(dst),
        ]
        subprocess.run(cmd_copy, check=True)



def load_qwen25vl(model_name: str, attn_impl: str = "sdpa"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def parse_shot_ranges(focal_shotsplit: List[int]) -> List[tuple[int, int]]:
    """ Convert split points to (start_frame, end_frame) ranges """
    ranges = []
    start = 0
    for end in focal_shotsplit:
        ranges.append((start, end))
        start = end + 1
    return ranges



@torch.inference_mode()
def infer_one(model, processor, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float, top_p: float) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(
        text=[text],
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

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return out_text.strip()


def inference_qwen25vl(
    *,
    model,
    processor,
    video_path: Path,
    video_info: Dict[str, Any],
    fps: int,
    max_pixels: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """
    Runs one inference using build_messages(...) and returns:
      out: raw model string output
      ok, total, selected: validation outputs from validate_selection(...)
    """
    messages, shot_durations, durations_str = build_messages(
        video_path,
        video_info,
        fps=fps,
        max_pixels=max_pixels,
    )


    #print(messages)

    
    out = infer_one(
        model=model,
        processor=processor,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    selected = parse_selected_indices(out)
    ok, total, selected = validate_selection(selected, shot_durations)

    return out, ok, total, selected

def main():
    import pandas as pd
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project_root",
        default="/projects/adsum-mllm-om",
        help="Absolute path to repo root that should be added to sys.path",
    )

    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--csv", required=True, help="Path to eval.csv")
    ap.add_argument("--row", type=int, default=0, help="Row number (0-based) from CSV to extract as input video")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--max_pixels", type=int, default=360 * 420)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--ffmpeg", default=FFMPEG_PATH, help="Path to ffmpeg binary")
    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"--project_root does not exist: {project_root}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    df = pd.read_csv((project_root / args.csv).resolve())
    if args.row < 0 or args.row >= len(df):
        raise IndexError(f"Row index {args.row} out of range for csv file {args.csv}")
    video_path = (project_root / Path(df.loc[args.row, "focal_relative_path"])).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
        
    video_info = df.loc[args.row, :].to_dict()    

    with tempfile.TemporaryDirectory() as td:
        model, processor = load_qwen25vl(args.model, attn_impl=args.attn_impl)

        out, ok, total, selected = inference_qwen25vl(
            model=model,
            processor=processor,
            video_path=video_path,
            video_info=video_info,
            fps=args.fps,
            max_pixels=args.max_pixels,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


    result_json_str = json.dumps(
        {
            "video": str(video_path.resolve()),
            "answers": out,
            # keeping these optional; remove if you truly don't want output changes
            "ok": ok,
            "total": total,
            "selected": selected,
        },
        ensure_ascii=False,
        indent=2,
    )

    payload = json.loads(result_json_str)
    print(result_json_str)
        

if __name__ == "__main__":
    main()
