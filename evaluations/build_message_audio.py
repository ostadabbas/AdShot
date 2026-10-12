"""
Advertisement Shot Selection Pipeline
--------------------------------------
Extracts frames + audio per shot from a 30-second ad video and builds
a structured multimodal prompt for an LLM to select a ~15-second clip.
"""

from __future__ import annotations

import ast
import json
import math
import shutil
import subprocess
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import cv2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FFMPEG_PATH  = "/projects/adsum-mllm-om/miniconda3/envs/shot/bin/ffmpeg"
FFPROBE_PATH = FFMPEG_PATH.replace("ffmpeg", "ffprobe")

# Percentile sample points keyed by number of frames-per-shot (k)
PCTS_BY_K: dict[int, list[float]] = {
    5: [0.10, 0.25, 0.50, 0.75, 0.90],
    4: [0.10, 0.25, 0.75, 0.90],
    3: [0.10, 0.50, 0.90],
    2: [0.25, 0.75],
    1: [0.50],
}

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def parse_shot_ranges(shotsplit: list[int]) -> list[tuple[int, int]]:
    """Convert a list of shot-end frame indices into (start, end) pairs."""
    ranges, start = [], 0
    for end in shotsplit:
        ranges.append((start, end))
        start = end + 1
    return ranges


def sample_percentiles(start: int, end: int, k: int = 5) -> list[int]:
    """
    Return k frame indices spread across [start, end] at the percentile
    positions defined in PCTS_BY_K.  k is clamped to [1, 5].
    """
    k = max(1, min(5, int(k)))
    if start == end:
        return [start] * k

    span = end - start
    return [
        max(start, min(end, int(math.floor(start + span * p + 0.5))))
        for p in PCTS_BY_K[k]
    ]


def _infer_video_id(video_path: Path, video_info: dict[str, Any]) -> str:
    """Return a stable identifier from the metadata row, falling back to filename stem."""
    for key in ("video_id", "id", "focal_video_id", "focal_id", "pair_id"):
        value = video_info.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return video_path.stem


def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds via OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return duration


def get_video_meta(video_path: Path) -> tuple[float, int, float]:
    """
    Return (fps, total_frames, duration_seconds).
    Tries decord first, falls back to OpenCV.
    """
    try:
        from decord import VideoReader, cpu  # type: ignore
        vr = VideoReader(str(video_path), ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        total = int(len(vr))
        if fps > 0 and total > 0:
            return fps, total, total / fps
    except Exception:
        pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or total <= 0:
        raise RuntimeError("Could not read video metadata. Install decord or opencv.")
    return fps, total, total / fps


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def _extract_audio(
    video_path: Path,
    out_path: Path,
    start_sec: float,
    dur_sec: float,
    ext: str,
    bitrate: str,
    sample_rate: Optional[int],
) -> Optional[str]:
    """
    Extract one audio segment with FFmpeg.
    Returns a file URI on success, None on failure.
    """
    if out_path.exists():
        return out_path.resolve().as_uri()

    cmd = [
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-fflags", "+discardcorrupt", "-err_detect", "ignore_err",
        "-ss", f"{start_sec:.6f}", "-t", f"{dur_sec:.6f}",
        "-i", str(video_path),
        "-vn", "-sn", "-dn", "-map", "0:a:0?",
    ]
    cmd += ["-c:a", "pcm_s16le"] if ext == "wav" else ["-c:a", "aac", "-b:a", bitrate]
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]
    cmd.append(str(out_path))

    try:
        subprocess.run(cmd, check=True)
        return out_path.resolve().as_uri()
    except subprocess.CalledProcessError:
        return None


def _extract_frame(video_path: Path, out_path: Path, t: float, duration: float) -> Path:
    """
    Seek to timestamp t and write a single JPEG frame.

    If ffmpeg produces no output (common when t is near the true end of the
    stream), the seek position is nudged back in 0.5 s steps up to 3 times
    before raising.  This handles the case where the container-reported
    duration is slightly longer than the decodable content.
    """
    def _run(seek: float) -> bool:
        out_path.unlink(missing_ok=True)
        result = subprocess.run(
            [
                FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
                "-fflags", "+discardcorrupt", "-err_detect", "ignore_err",
                "-ss", f"{seek:.6f}", "-i", str(video_path),
                "-frames:v", "1", "-strict", "unofficial",
                "-pix_fmt", "yuvj420p", "-q:v", "2",
                str(out_path),
            ],
            check=False,   # don't raise — we check the file ourselves
        )
        return out_path.exists() and out_path.stat().st_size > 0

    MAX_RETRIES  = 3
    NUDGE_STEP   = 0.5   # seconds to step back on each retry

    seek = min(t, max(0.0, duration - NUDGE_STEP))
    for attempt in range(MAX_RETRIES + 1):
        if _run(seek):
            return out_path
        nudged = seek - NUDGE_STEP * (attempt + 1)
        if nudged < 0:
            break
        print(f"  [frame] no output at t={seek:.3f}s, retrying at t={nudged:.3f}s")
        seek = nudged

    raise RuntimeError(
        f"ffmpeg produced no output for {video_path.name} "
        f"near t={t:.3f}s (duration reported as {duration:.3f}s)"
    )


# ---------------------------------------------------------------------------
# Per-shot extraction
# ---------------------------------------------------------------------------

def _process_shot(
    shot_idx: int,
    start_frame: int,
    end_frame: int,
    video_path: Path,
    fps: float,
    ad_len: float,
    k: int,
    out_base: Path,
    extract_audio: bool,
    audio_ext: str,
    audio_bitrate: str,
    audio_sample_rate: Optional[int],
) -> tuple[int, list[str], float, Optional[str]]:
    """Extract k frames and one audio clip for a single shot."""
    dur_sec   = (end_frame - start_frame + 1) / fps
    start_sec = start_frame / fps
    out_dir   = out_base / f"shot_{shot_idx:02d}"
    out_dir.mkdir(exist_ok=True)

    # --- audio ---
    audio_uri = None
    if extract_audio:
        audio_uri = _extract_audio(
            video_path,
            out_dir / f"audio.{audio_ext.lower()}",
            start_sec, dur_sec, audio_ext.lower(),
            audio_bitrate, audio_sample_rate,
        )

    # --- frames ---
    frame_indices = sample_percentiles(start_frame, end_frame, k)
    unique_indices = list(dict.fromkeys(frame_indices))   # preserve order, deduplicate

    final_paths = [out_dir / f"frame_{j:02d}.jpg" for j in range(1, len(frame_indices) + 1)]

    if not all(p.exists() for p in final_paths):
        extracted: dict[int, Path] = {}
        for fidx in unique_indices:
            # Stay at least 1.5 frame-periods inside the stream to avoid
            # seeking past decodable content (container duration can be stale)
            t = max(0.0, min(fidx / fps, ad_len - 1.5 / fps))
            # print(f"[Shot {shot_idx}] fidx={fidx}, t={t:.6f}s  (ad_len={ad_len:.3f}s)")
            tmp = out_dir / f"frame_tmp_{fidx}.jpg"
            extracted[fidx] = _extract_frame(video_path, tmp, t, ad_len)

        for j, fidx in enumerate(frame_indices, 1):
            shutil.copy2(extracted[fidx], out_dir / f"frame_{j:02d}.jpg")

        for tmp in extracted.values():
            tmp.unlink(missing_ok=True)

    uris = [p.resolve().as_uri() for p in final_paths]
    return shot_idx, uris, dur_sec, audio_uri


# ---------------------------------------------------------------------------
# Public extraction entry-point
# ---------------------------------------------------------------------------

def extract_kframes_per_shot(
    video_path: Path,
    video_info: dict[str, Any],
    output_dir: Optional[Path] = None,
    k: int = 5,
    extract_audio: bool = True,
    audio_ext: str = "m4a",
    audio_bitrate: str = "128k",
    audio_sample_rate: Optional[int] = None,
    max_workers: int = 8,
) -> tuple[list[list[str]], list[float], list[Optional[str]]]:
    """
    For each shot in the video, extract k frames and one audio segment.

    Returns:
        frame_uris_by_shot  — list[list[str]]        (one URI list per shot)
        shot_durations      — list[float]             (seconds per shot)
        audio_uris_by_shot  — list[Optional[str]]     (one URI or None per shot)
    """
    output_dir = output_dir or Path(tempfile.mkdtemp())
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = _infer_video_id(video_path, video_info)
    out_base  = output_dir / f"{video_id}_k={k}"
    out_base.mkdir(parents=True, exist_ok=True)

    shotsplit = video_info["focal_shotsplit"]
    if isinstance(shotsplit, str):
        shotsplit = ast.literal_eval(shotsplit)

    ad_len = get_video_duration(video_path)
    fps    = (shotsplit[-1] + 1) / ad_len
    ranges = parse_shot_ranges(shotsplit)

    shot_args = [
        (i, s, e, video_path, fps, ad_len, k, out_base,
         extract_audio, audio_ext, audio_bitrate, audio_sample_rate)
        for i, (s, e) in enumerate(ranges, 1)
    ]

    results: dict[int, tuple[list[str], float, Optional[str]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_shot, *args): args[0] for args in shot_args}
        for future in as_completed(futures):
            idx, uris, dur, audio = future.result()
            results[idx] = (uris, dur, audio)

    n = len(ranges)
    return (
        [results[i][0] for i in range(1, n + 1)],
        [results[i][1] for i in range(1, n + 1)],
        [results[i][2] for i in range(1, n + 1)],
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
TASK: Advertisement Shot Selection

You are an expert in advertisement clipping. Below is a 30-second video \
advertisement split into shots. Your task is to select a subset of whole \
shots whose total duration is approximately 15 seconds.

INPUT FORMAT:
- Each shot is represented by {k} frames sampled at {pcts} of each shot's duration
- Each shot also includes an audio segment (if available)
- Shots are labeled [Shot 1], [Shot 2], etc. in order

OUTPUT FORMAT:
Return a JSON object with exactly these three fields:
{{
  "selected_shots":   ["shotX", "shotY", ...],
  "total_length_sec": XX.XX,
  "explanations":     ["reason for shotX", "reason for shotY", ...]
}}

TASK STEPS:
1. Review all shots and their durations provided below
2. Analyse the visual content (and audio, if present) of each shot
3. Select shots that together create an effective ~15-second clip
4. Sum the durations of all selected shots
5. Verify the total is between 14 and 16 seconds
6. Write a brief explanation for each selected shot

REQUIREMENTS:
- Select whole shots only (no partial shots)
- total_length_sec MUST equal your calculated sum
- Total duration MUST be between 14–16 seconds (target: 15 s)
- Return ONLY valid JSON with double quotes, no extra text
"""


def build_messages(
    video_path: Path,
    video_info: dict[str, Any],
    k: int,
    max_pixels: Optional[int],
    frame_cache_dir: Optional[Path],
) -> tuple[list[dict[str, Any]], list[float], str]:
    """
    Build the multimodal message list for the LLM.

    Returns:
        messages        — OpenAI-style message list (system + user)
        shot_durations  — duration in seconds for each shot
        durations_str   — human-readable summary of shot durations
    """
    _, _, _ = get_video_meta(video_path)  # validates the file early

    pcts_str = ", ".join(f"{int(p * 100)}%" for p in PCTS_BY_K[k])
    system_text = SYSTEM_PROMPT.format(k=k, pcts=pcts_str)

    # t0 = time.perf_counter()
    frame_uris_by_shot, shot_durations, audio_uris_by_shot = extract_kframes_per_shot(
        video_path, video_info, k=k, output_dir=frame_cache_dir
    )
    # print(f"extract_kframes_per_shot took {time.perf_counter() - t0:.3f}s")

    n_shots       = len(shot_durations)
    durations_str = "; ".join(f"shot {i}: {d:.2f}s" for i, d in enumerate(shot_durations, 1))
    durations_map = {f"shot{i}": round(d, 2) for i, d in enumerate(shot_durations, 1)}

    # --- user content ---
    content: list[dict[str, Any]] = [
        {"type": "text", "text": (
            f"This advertisement contains {n_shots} shots.\n"
            f"Shot durations (seconds): {json.dumps(durations_map)}\n\n"
            "The audio and frames of each shot follow:\n"
        )},
    ]

    for i, (audio_uri, frame_uris) in enumerate(zip(audio_uris_by_shot, frame_uris_by_shot), 1):
        content.append({"type": "text", "text": f"[Shot {i}]"})
        if audio_uri is not None:
            content.append({"type": "audio", "audio": audio_uri})
        for uri in frame_uris:
            item: dict[str, Any] = {"type": "image", "image": uri}
            if max_pixels is not None:
                item["max_pixels"] = max_pixels
            content.append(item)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user",   "content": content},
    ]
    return messages, shot_durations, durations_str


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    import sys
    from time import perf_counter

    import pandas as pd
    from tqdm import tqdm

    ap = argparse.ArgumentParser(description="Build shot-selection prompts for ad videos.")
    ap.add_argument("--project_root",    default="/projects/adsum-mllm-om")
    ap.add_argument("--csv",             required=True, help="Path to CSV file")
    ap.add_argument("--fps",             type=int, default=5, help="Frames per shot (1–5)")
    ap.add_argument("--max_pixels",      type=int, default=360 * 420)
    ap.add_argument("--frame_cache_dir", default="frame_cache")
    ap.add_argument("--out",             default="processing.log")
    ap.add_argument("--row",             type=int, default=None, help="Process only this row index")
    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (project_root / path).resolve()

    csv_path  = _resolve(args.csv)
    out_path  = _resolve(args.out)
    cache_dir = _resolve(args.frame_cache_dir)

    df = pd.read_csv(csv_path)
    if args.row is not None:
        df = df.iloc[[args.row]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    with out_path.open("w", encoding="utf-8") as log:
        log.write(f"Processing Log — {pd.Timestamp.now()}\n")
        log.write(f"CSV: {csv_path}\n{'=' * 80}\n\n")

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
            video_info = row.to_dict()
            p          = Path(video_info["focal_relative_path"])
            video_path = (p if p.is_absolute() else project_root / p).expanduser().resolve()

            try:
                if not video_path.exists():
                    raise FileNotFoundError(f"Video not found: {video_path}")

                t0 = perf_counter()
                messages, shot_durations, durations_str = build_messages(
                    video_path=video_path,
                    video_info=video_info,
                    k=args.fps,
                    max_pixels=args.max_pixels,
                    frame_cache_dir=cache_dir,
                )
                print(f"[Row {i}] build_messages: {perf_counter() - t0:.3f}s")

                log.write(f"Row {i}: OK — {video_path.name}\n")
                ok += 1

            except Exception as e:
                msg = f"Row {i}: FAIL — {type(e).__name__}: {e}\n{traceback.format_exc()}\n"
                log.write(msg)
                print(msg)
                fail += 1

            log.flush()

        log.write(f"\n{'=' * 80}\n")
        log.write(f"Summary: {ok} OK, {fail} FAIL, {ok + fail} Total\n")
        if ok + fail > 0:
            log.write(f"Success rate: {ok / (ok + fail) * 100:.2f}%\n")

    print(f"\nLog saved to: {out_path}")
    print(f"Summary: {ok} OK, {fail} FAIL")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    main()
