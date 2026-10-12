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
from transformers import AutoModel, AutoTokenizer
import json
import re
import ast
from typing import Dict, Any, List, Optional
import itertools
from PIL import Image
from urllib.parse import urlparse, unquote

from build_message import build_messages
from seed_all import set_seed

set_seed()
# >>> YOUR ENVIRONMENT <<<
FFMPEG_PATH = "/projects/adsum-mllm-om/miniconda3/envs/shot/bin/ffmpeg"
os.environ["PATH"] = "/projects/adsum-mllm-om/miniconda3/envs/shot/bin:" + os.environ["PATH"]

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
    # extract JSON object from possible surrounding text
    m = re.search(r"\{.*\}", out, flags=re.DOTALL)
    if not m:
        return []
    obj = json.loads(m.group(0))

    shots = obj.get("selected_shots", [])
    # ["shot4", "shot17"] -> [4, 17]
    return [int(re.search(r"\d+", s).group()) for s in shots if re.search(r"\d+", s)]

def file_uri_to_path(uri: str) -> str:
    uri = uri.replace("\n", "").replace("\r", "")  # fix wrapped paths
    p = urlparse(uri)
    if p.scheme != "file":
        raise ValueError(f"Expected file:// URI, got: {uri}")
    return unquote(p.path)

def to_minicpm_msgs(raw_msgs):
    """
    raw_msgs: your list of dicts like [{'role':'system','content':[...]} , {'role':'user','content':[...]}]

    Returns: msgs for MiniCPM: [{'role':'user','content':[...]}]
    """
    system_text = ""
    user_parts = []

    for m in raw_msgs:
        role = m["role"]
        for item in m.get("content", []):
            t = item.get("type")

            if role == "system" and t == "text":
                system_text += item.get("text", "")
                continue

            if role == "user":
                if t == "text":
                    user_parts.append(item.get("text", ""))
                elif t == "image":
                    img_path = file_uri_to_path(item["image"])
                    user_parts.append(Image.open(img_path).convert("RGB"))
    # MiniCPM doesn't need a separate system role; prepend it as plain text
    if system_text.strip():
        user_parts.insert(0, system_text.strip())
    return [{"role": "user", "content": user_parts}]


@torch.inference_mode()
def infer_one(model, tokenizer, messages, max_new_tokens: int, temperature: float, top_p: float,
             max_inp_length: int = 32768) -> str:

    msgs = to_minicpm_msgs(messages)

    do_sample = temperature > 0.0
    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        stream=True,

        # IMPORTANT: avoid processor truncation + reduce image token explosion
        max_inp_length=max_inp_length,
        max_slice_nums=1,
        use_image_id=False,

        # generation controls
        max_new_tokens=max_new_tokens,
        sampling=True,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
    )

    return "".join(answer)


def inference_minicpm(
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
        tokenizer=processor,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    selected = parse_selected_indices(out)
    ok, total, selected = validate_selection(selected, shot_durations)
    return out, ok, total, selected

def load_minicpm(model_name: str, attn_impl: str = "sdpa"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model = model.eval().cuda()    
    return model, tokenizer

def main():
    import pandas as pd
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project_root",
        default="/projects/adsum-mllm-om",
        help="Absolute path to repo root that should be added to sys.path",
    )

    ap.add_argument("--model", default="openbmb/MiniCPM-V-4_5")
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
        model, processor = load_minicpm(args.model, attn_impl=args.attn_impl)

        out, ok, total, selected = inference_minicpm(
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
