from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
import subprocess
import ast
import json
import shutil
import math


FFMPEG_PATH = "/projects/adsum-mllm-om/miniconda3/envs/shot/bin/ffmpeg"
BASE_PCTS = [0.10, 0.25, 0.50, 0.75, 0.90]

# Subsets chosen from the same 5 points
PCTS_BY_K = {
    5: [0.10, 0.25, 0.50, 0.75, 0.90],
    4: [0.10, 0.25, 0.75, 0.90],
    3: [0.10, 0.50, 0.90],
    2: [0.25, 0.75], #wen, revised this
    1: [0.50],
}

def parse_shot_ranges(focal_shotsplit: List[int]) -> List[tuple[int, int]]:
    ranges, start = [], 0
    for end in focal_shotsplit:
        ranges.append((start, end))
        start = end + 1
    return ranges

def sample_percentiles(start: int, end: int, k: int = 5) -> list[int]:
    # inclusive range [start, end]
    if end < start:
        raise ValueError(f"Invalid range: {start}..{end}")

    k = max(1, min(5, int(k)))
    if start == end:
        return [start] * k

    span = end - start  # inclusive span - 1
    pcts = PCTS_BY_K[k]

    idxs = []
    for p in pcts:
        # round-half-up (avoid Python banker rounding)
        x = start + span * p
        idx = int(math.floor(x + 0.5))
        idx = max(start, min(end, idx))
        idxs.append(idx)

    return idxs
    
def extract_kframes_per_shot(
    video_path: Path,
    video_info: Dict[str, Any],
    output_dir: Optional[Path] = None,
    k: int = 5,
) -> Tuple[List[List[str]], List[float]]:
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    output_dir.mkdir(parents=True, exist_ok=True)

    shotsplit = video_info["focal_shotsplit"]
    ad_len = video_info["focal_ad_length"]
    if isinstance(shotsplit, str): shotsplit = ast.literal_eval(shotsplit)
    if isinstance(ad_len, str): ad_len = float(ad_len)

    fps = (shotsplit[-1] + 1) / ad_len
    ranges = parse_shot_ranges(shotsplit)

    def sample(start: int, end: int) -> List[int]:
        n = end - start + 1
        if n <= 1: return [start] * k
        if n < k: return list(range(start, end + 1)) + [end] * (k - n)
        return [round(start + (n - 1) * j / (k - 1)) for j in range(k)]

    frame_uris_by_shot, shot_durs = [], []

    for i, (s, e) in enumerate(ranges, 1):
        shot_durs.append((e - s + 1) / fps)
        out_dir = output_dir / f"shot_{i:02d}"
        out_dir.mkdir(exist_ok=True)

        frame_indices = sample_percentiles(s, e, k)
        unique_indices = []
        seen = set()
        for idx in frame_indices:
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)

        # Extract unique frames
        extracted_frames = {}
        for fidx in unique_indices:
            t = fidx / fps
            out = out_dir / f"frame_tmp_{fidx}.jpg"
            subprocess.run(
                [
                    FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
                    "-fflags", "+discardcorrupt",
                    "-err_detect", "ignore_err",
                    "-ss", f"{t:.6f}", "-i", str(video_path),
                    "-frames:v", "1",
                    "-strict", "unofficial",
                    "-pix_fmt", "yuvj420p",
                    "-q:v", "2",
                    str(out),
                ],
                check=True,
            )
            extracted_frames[fidx] = out

        # Create final frames (copying duplicates)
        uris = []
        for j, fidx in enumerate(frame_indices, 1):
            final_out = out_dir / f"frame_{j:02d}.jpg"
            shutil.copy2(extracted_frames[fidx], final_out)
            uris.append(final_out.resolve().as_uri())
        
        # Clean up temporary files
        for temp_file in extracted_frames.values():
            temp_file.unlink()

        frame_uris_by_shot.append(uris)
    return frame_uris_by_shot, shot_durs

def get_video_meta(video_path: Path) -> Tuple[float, int, float]:
    """
    Get video metadata: FPS, total frames, and duration.
    
    Returns:
        Tuple[float, int, float]: (fps, total_frames, duration_seconds)
    """
    # Prefer decord (already implied by qwen-vl-utils[decord])
    try:
        from decord import VideoReader, cpu  # type: ignore
        vr = VideoReader(str(video_path), ctx=cpu(0))
        orig_fps = float(vr.get_avg_fps())
        total_frames = int(len(vr))
        if orig_fps > 0 and total_frames > 0:
            duration = total_frames / orig_fps
            return orig_fps, total_frames, duration
    except Exception:
        pass

    # Fallback OpenCV
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open video")
        orig_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if orig_fps > 0 and total_frames > 0:
            duration = total_frames / orig_fps
            return orig_fps, total_frames, duration
    except Exception as e:
        raise RuntimeError("Could not read video metadata. Install decord/opencv.") from e


def build_messages(
    video_path: Path,
    video_info: Dict[str, Any],
    fps: int,
    max_pixels: Optional[int],
) -> tuple[list[dict[str, Any]], list[float], str]:
    
    orig_fps, total_frames, duration = get_video_meta(video_path)
    pcts_str = ", ".join(f"{int(p * 100)}%" for p in PCTS_BY_K[fps])
    
    # 1) Extract frames per shot (URIs) + durations
    frame_uris_by_shot, shot_durations = extract_kframes_per_shot(video_path, video_info, k=fps)
    n_shots = len(shot_durations)
    durations_str = "; ".join([f"shot {i}: {d:.2f}s" for i, d in enumerate(shot_durations, start=1)])
    
    # 2) Build single comprehensive user prompt
    system_prompt = (
        f"TASK: Advertisement Shot Selection\n\n"
        f"You are an expert in advertisement clipping. Below is a 30-second video advertisement split into shots. "
        f"Your task is to select a subset of whole shots whose total duration is approximately 15 seconds.\n\n"
        
        f"INPUT FORMAT:\n"
        f"- Each shot is represented by {fps} frames sampled at {pcts_str} of each shot's duration\n"
        f"- Shots are labeled [Shot 1], [Shot 2], etc. with frames shown in order\n\n"
        
        "OUTPUT FORMAT:\n"
        "Return a JSON object with exactly these three fields:"
        "{\n"
        "  'selected_shots': ['shotX', 'shotY', 'shotZ',...],  // Array of shot IDs\n"
        "  'total_length_sec': XX.XX,                  // Sum of selected durations\n"
        "  'explanations': ['reason of selecting shotX', 'reason  of selecting shotY', 'reason  of selecting shotZ',...] // One per selected shot\n"
        "}\n\n"
        "Example 1"
        "{\n"
        "  'selected_shots': ['shot4', 'shot7', 'shot8', 'shot15'],\n"
        "  'total_length_sec': 14.76,\n"
        "  'explanations': ['reason of selecting shot4', 'reason  of selecting shot7', 'reason  of selecting shot8','reason  of selecting shot15']\n"
        "}\n\n"
        "Example 2"
        "{\n"
        "  'selected_shots': ['shot2', 'shot6', 'shot10'],\n"
        "  'total_length_sec': 15.23,\n"
        "  'explanations': ['reason of selecting shot2', 'reason  of selecting shot6', 'reason  of selecting shot10',]\n"
        "}\n\n"

        
        "TASK STEPS:\n"
        "1. Review all shots and their durations provided below\n"
        "2. Analyze the visual content of each shot\n"
        "3. Select shots that together create an effective ~15 second clip\n"
        "4. Calculate total: sum the durations of all shots you selected\n"
        "5. Verify your total is between 14-16 seconds\n"
        "6. Write brief explanation for why each shot was selected\n\n"
        
        "REQUIREMENTS:\n"
        "- Select whole shots only (no partial shots)\n"
        "- Total duration MUST be between 14-16 seconds (target: 15 seconds)\n"
        "- total_length_sec MUST equal your actual calculated sum\n"
        "- Return ONLY valid JSON with double quotes, no extra text\n\n"
    )
    
    # 3) Build content: user_prompt first, then all shots with images
    content: List[Dict[str, Any]] = []
    user_prompt = (
        f"This advertisement contains {n_shots} shots. The duration of each shot (in seconds) is: "
        + json.dumps({f"shot{i}": round(d, 2) for i, d in enumerate(shot_durations, start=1)})
        + "\n"
        + "The frames of each shot are below: \n"
    )
    content.append({"type": "text", "text": user_prompt})
    
    for i, frame_uris in enumerate(frame_uris_by_shot, start=1):
        content.append({"type": "text", "text": f"[Shot {i}]"})
        for uri in frame_uris:
            item = {"type": "image", "image": uri}
            if max_pixels is not None:
                item["max_pixels"] = max_pixels
            content.append(item)
   
    # 4) Return
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": content,
        },
    ], shot_durations, durations_str
    

def main():
    import argparse, sys
    from pathlib import Path
    import pandas as pd
    from tqdm import tqdm
    from time import perf_counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default="/projects/adsum-mllm-om")
    ap.add_argument("--csv", required=True, help='Path to CSV file')
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--max_pixels", type=int, default=360 * 420)
    ap.add_argument("--out", default="processing.log", help='Output log file')
    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    csv_path = Path(args.csv)
    csv_path = (project_root / csv_path).resolve() if not csv_path.is_absolute() else csv_path.expanduser().resolve()
    df = pd.read_csv(csv_path)

    out_path = Path(args.out)
    out_path = (project_root / out_path).resolve() if not out_path.is_absolute() else out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Processing Log - {pd.Timestamp.now()}\n")
        f.write(f"CSV: {csv_path}\n")
        f.write("=" * 80 + "\n\n")
        
        pbar = tqdm(df.iterrows(), total=len(df), desc="Processing videos")
        for i, row in pbar:
            video_info = row.to_dict()
            p = Path(video_info["focal_relative_path"])
            video_path = (p if p.is_absolute() else project_root / p).expanduser().resolve()

            try:
                if not video_path.exists():
                    raise FileNotFoundError(f"Video not found: {video_path}")
                
                t0 = perf_counter()
                messages, shot_durations, durations_str = build_messages(
                    video_path=video_path,
                    video_info=video_info,
                    fps=args.fps,
                    max_pixels=args.max_pixels,
                )
                dt = perf_counter() - t0
                print(messages)
                print(f"[Row {i}] build_messages time: {dt:.3f}s")

                
                print("=== system_prompt ===\n", next(p["text"] for p in messages[0]["content"] if p["type"]=="text"))
                print("=== task_info ===\n",     [p["text"] for p in next(m for m in messages if m["role"]=="user")["content"] if p["type"]=="text"][-1])

                f.write(f"Row {i}: OK - {video_path.name}\n")
                f.flush()
                ok += 1
                
                
            except Exception as e:
                f.write(f"Row {i}: FAIL - {type(e).__name__}: {e}\n")
                f.flush()
                fail += 1

            pbar.set_postfix(ok=ok, fail=fail)

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Summary: {ok} OK, {fail} FAIL, {ok+fail} Total\n")
        if ok + fail > 0:
            f.write(f"Success rate: {ok/(ok+fail)*100:.2f}%\n")

    print(f"\nLog saved to: {out_path}")
    print(f"Summary: {ok} OK, {fail} FAIL")
    
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    main()