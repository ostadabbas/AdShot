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
from transformers import AutoProcessor, AutoModelForImageTextToText
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

MolmoImageType = Image.Image

MolmoImageType = Image.Image  # what molmo expects in the "image" field

def resize_for_molmo(img: Image.Image, processor=None) -> Image.Image:
    """
    Resize to Molmo2 processor's configured size (default 378x378) and ensure RGB.
    - If `processor` is provided, we read size from processor.image_processor.size.
    - Uses BILINEAR; matches config resample=2 (PIL bilinear-ish). :contentReference[oaicite:2]{index=2}
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    target: Tuple[int, int] = (378, 378)
    if processor is not None and hasattr(processor, "image_processor"):
        sz = getattr(processor.image_processor, "size", None)
        if isinstance(sz, dict) and "width" in sz and "height" in sz:
            target = (int(sz["width"]), int(sz["height"]))

    if img.size != target:
        img = img.resize(target, resample=Image.BILINEAR)

    return img


def _load_pil_image(
    image_ref: Union[str, MolmoImageType],
    timeout: int = 20,
) -> MolmoImageType:
    """
    Accepts:
      - PIL.Image.Image -> returned as-is
      - "file:///abs/path.jpg" -> loaded from disk
      - "/abs/path.jpg" or "rel/path.jpg" -> loaded from disk
      - "http(s)://..." -> downloaded then loaded
    """
    if isinstance(image_ref, Image.Image):
        return image_ref

    if not isinstance(image_ref, str):
        raise TypeError(f"Unsupported image ref type: {type(image_ref)}")

    s = image_ref.strip()
    if s.startswith("file://"):
        path = urlparse(s).path
        if not path:
            raise ValueError(f"Bad file URI: {s}")
        return Image.open(path).convert("RGB")

    if s.startswith("http://") or s.startswith("https://"):
        resp = requests.get(s, stream=True, timeout=timeout)
        resp.raise_for_status()
        return Image.open(resp.raw).convert("RGB")

    # Assume local path
    if not os.path.exists(s):
        raise FileNotFoundError(f"Image path not found: {s}")
    return Image.open(s).convert("RGB")

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

import re
import json
import ast
from typing import List

def parse_selected_indices(out: str) -> List[int]:
    """
    Robust for model outputs that:
      - contain truncated/invalid JSON
      - wrap JSON in extra text
      - use either double-quotes (JSON) or single-quotes (python-ish dict)
      - output selected_shots as ["shot4", ...] or ["4", ...] or [4, ...]
    Returns: [4, 17, ...]
    """

    def _nums_from_text(s: str) -> List[int]:
        # Prefer explicit "shot<number>" occurrences
        hits = re.findall(r"\bshot\s*0*(\d+)\b", s, flags=re.IGNORECASE)
        if hits:
            return [int(x) for x in hits]
        # Fallback: any integers
        return [int(x) for x in re.findall(r"-?\d+", s)]

    # 1) Fast-path: if we can locate the selected_shots array, parse it even if JSON is truncated
    m = re.search(r'"selected_shots"\s*:\s*\[(.*?)\]', out, flags=re.DOTALL)
    if not m:
        m = re.search(r"'selected_shots'\s*:\s*\[(.*?)\]", out, flags=re.DOTALL)

    if m:
        inner = m.group(1)
        # Extract entries that look like shot ids / numbers
        # e.g. "shot12", 'shot12', shot12, 12
        tokens = re.findall(r'["\']?\s*(?:shot\s*)?0*(\d+)\s*["\']?', inner, flags=re.IGNORECASE)
        if tokens:
            # de-dupe while preserving order
            out_nums, seen = [], set()
            for t in tokens:
                n = int(t)
                if n not in seen:
                    out_nums.append(n); seen.add(n)
            return out_nums

    # 2) Try full JSON object decode if present
    mj = re.search(r"\{.*\}", out, flags=re.DOTALL)
    if mj:
        blob = mj.group(0)
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(blob)
                shots = obj.get("selected_shots", [])
                nums, seen = [], set()
                for s in shots:
                    if isinstance(s, (int, float)) and int(s) == s:
                        n = int(s)
                    else:
                        ms = re.search(r"\d+", str(s))
                        if not ms:
                            continue
                        n = int(ms.group(0))
                    if n not in seen:
                        nums.append(n); seen.add(n)
                if nums:
                    return nums
            except Exception:
                pass

    # 3) Last resort: regex scan entire output
    return _nums_from_text(out)

def file_uri_to_path(uri: str) -> str:
    uri = uri.replace("\n", "").replace("\r", "")  # fix wrapped paths
    p = urlparse(uri)
    if p.scheme != "file":
        raise ValueError(f"Expected file:// URI, got: {uri}")
    return unquote(p.path)

def convert_to_molmo_messages(
    messages: List[Dict[str, Any]],
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    """
    Take input in your 2-message format (system + user),
    merge system.content + user.content into ONE user message,
    and load images as PIL.Image.Image for Molmo.

    Output:
      [{"role": "user", "content": [...merged blocks...]}]
    """
    system_blocks: List[Dict[str, Any]] = []
    user_blocks: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", [])

        molmo_content: List[Dict[str, Any]] = []
        for block in content:
            btype = block.get("type")

            if btype == "text":
                molmo_content.append({"type": "text", "text": block.get("text", "")})

            elif btype == "image":
                pil_img = _load_pil_image(block.get("image"), timeout=timeout)
                pil_img = resize_for_molmo(pil_img, processor=None)
                molmo_content.append({"type": "image", "image": pil_img})

            else:
                raise ValueError(f"Unsupported content block type: {btype}")

        if role == "system":
            system_blocks.extend(molmo_content)
        elif role == "user":
            user_blocks.extend(molmo_content)
        else:
            # ignore other roles
            continue

    merged = system_blocks + user_blocks
    return [{"role": "user", "content": merged}]


@torch.inference_mode()
def infer_one(model, processor, messages, max_new_tokens: int, temperature: float, top_p: float,
             max_inp_length: int = 32768) -> str:

    msgs = convert_to_molmo_messages(messages)
    inputs = processor.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0.0
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=448)

    # only get generated tokens; decode them to text
    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(
        generated_tokens, 
        skip_special_tokens=True
    )

    return generated_text


def inference_molmo2(
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

def load_molmo2(model_name: str, attn_impl: str = "sdpa"):
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
        attn_implementation=attn_impl
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
    ).eval() 
    return model, processor

def main():
    import pandas as pd
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project_root",
        default="/projects/adsum-mllm-om",
        help="Absolute path to repo root that should be added to sys.path",
    )

    ap.add_argument("--model", default="allenai/Molmo2-8B")
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
        model, processor = load_molmo2(args.model, attn_impl=args.attn_impl)

        out, ok, total, selected = inference_molmo2(
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
