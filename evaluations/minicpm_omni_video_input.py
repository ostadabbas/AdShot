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
import soundfile as sf
import itertools
from PIL import Image
import librosa

from build_message_audio import build_messages

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



def load_minicpmo26(model_name: str, attn_impl: str = "sdpa"):
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        init_vision=True,
        init_audio=True,
        init_tts=False,
    )
    model = model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

def parse_shot_ranges(focal_shotsplit: List[int]) -> List[tuple[int, int]]:
    """ Convert split points to (start_frame, end_frame) ranges """
    ranges = []
    start = 0
    for end in focal_shotsplit:
        ranges.append((start, end))
        start = end + 1
    return ranges

# def convert_qwen_to_minicpm(qwen_messages):
#     """
#     Convert Qwen-style messages to a flat MiniCPM-o content list.
#     """
#     content = []

#     for msg in qwen_messages:
#         items = msg["content"]

#         # If content is a plain list of strings (like the system intro)
#         if isinstance(items, list) and all(isinstance(i, str) for i in items):
#             for text in items:
#                 content.append(text)
#             continue

#         # If content is a list of dicts (Qwen format)
#         for item in items:
#             if isinstance(item, str):
#                 content.append(item)
#             elif isinstance(item, dict):
#                 if item["type"] == "text":
#                     content.append(item["text"])
#                 elif item["type"] == "image":
#                     path = item["image"].replace("file://", "")
#                     img = Image.open(path).convert("RGB")
#                     content.append(img)
#                 elif item["type"] == "audio":
#                     path = item["audio"].replace("file://", "")
#                     audio_np, _ = librosa.load(path, sr=16000, mono=True)
#                     content.append(audio_np)

#     return content

def convert_qwen_to_minicpm(qwen_messages):
    """
    Convert Qwen-style messages to a flat MiniCPM-o content list.
    Each shot's audio is split evenly across its frames, with <unit> markers.
    """
    content = []

    for msg in qwen_messages:
        items = msg["content"]

        # If content is a plain list of strings (like the system intro)
        if isinstance(items, list) and all(isinstance(i, str) for i in items):
            for text in items:
                content.append(text)
            continue

        # Collect items per shot, then structure them
        current_audio = None
        current_frames = []

        for item in items:
            if isinstance(item, str):
                # Flush previous shot if any
                if current_frames:
                    _flush_shot(content, current_audio, current_frames)
                    current_audio = None
                    current_frames = []
                content.append(item)

            elif isinstance(item, dict):
                if item["type"] == "text":
                    # Flush previous shot if any
                    if current_frames:
                        _flush_shot(content, current_audio, current_frames)
                        current_audio = None
                        current_frames = []
                    content.append(item["text"])

                elif item["type"] == "audio":
                    path = item["audio"].replace("file://", "")
                    current_audio, _ = librosa.load(path, sr=16000, mono=True)

                elif item["type"] == "image":
                    path = item["image"].replace("file://", "")
                    img = Image.open(path).convert("RGB")
                    current_frames.append(img)

        # Flush last shot
        if current_frames:
            _flush_shot(content, current_audio, current_frames)

    return content


def _flush_shot(content, audio, frames):
    n = len(frames)
    min_samples = 16000  # 1 second minimum per chunk
    
    for i, frame in enumerate(frames):
        content.append("<unit>")
        content.append(frame)
        if audio is not None:
            chunk_len = len(audio) // n
            start = chunk_len * i
            end = chunk_len * (i + 1) if i < n - 1 else len(audio)
            chunk = audio[start:end]
            # Pad if too short
            if len(chunk) < min_samples:
                chunk = np.pad(chunk, (0, min_samples - len(chunk)), mode='constant')
            content.append(chunk)

@torch.inference_mode()
def infer_one(model, tokenizer, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float, top_p: float) -> str:
    content = convert_qwen_to_minicpm(messages)
    user_msg = {"role": "user", "content": content}

    sys_msg = model.get_sys_prompt(mode="omni", language='en')
    msgs = [sys_msg, user_msg]
    use_sampling = temperature > 0.0

    out = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=use_sampling,
        temperature=temperature if use_sampling else 1.0,
        max_new_tokens=max_new_tokens,
        omni_input=True,
        use_tts_template=False,
        generate_audio=False,
        max_slice_nums=1,
        use_image_id=False,
        return_dict=True,
    )['text']
    return out.strip()

def inference_minicpmo26(
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

def main():
    import pandas as pd
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project_root",
        default="/projects/adsum-mllm-om",
        help="Absolute path to repo root that should be added to sys.path",
    )
    ap.add_argument("--model", default="openbmb/MiniCPM-o-2_6")
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
        print("Loop")
        model, processor = load_minicpmo26(args.model, attn_impl=args.attn_impl)

        out, ok, total, selected = inference_minicpmo26(
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
