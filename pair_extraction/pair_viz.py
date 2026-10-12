"""
Testing/QA viz: for each chosen pair in a fullmatch CSV, render a 2-row montage.

Layout (one PNG per pair):
  - One column per shot of the SHORT (target) video, in shot order.
  - Top cell    = mid frame (p50 keyframe) of that short-video shot.
  - Bottom cell = mid frame of the corresponding LONG (focal) video shot that
                  maps to it (per the `mapping` column). Blank if unmatched.

The `mapping` column is {focal_long_shot: target_short_shot|None}; we invert it
to short_shot -> focal_shot to drive the columns by the short video.

Frames are decoded DIRECTLY from the source videos (no keyframes dir needed --
the production pipeline does not write keyframe JPEGs). The p50 frame index is
computed exactly as s1_keyframes.py does, so the rendered frame matches what the
keyframe stage would have saved. Everything needed comes from the fullmatch CSV:
focal/target shotsplit + relative_path columns.

Usage:
  python pair_viz.py --fullmatch_csv ... --output_dir ...
                     [--video_dir ...] [--percentile 50] [--brand 105]
"""
import argparse
import ast
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config
from s1_keyframes import KeyframeExtractor

CELL_W, CELL_H = 240, 135          # 16:9 thumbnail per frame
LABEL_H = 18                        # text strip above each frame
PAD = 6
HEADER_H = 30

# One decoder (320x180, percentiles) + a per-video full-decode cache. The same
# focal video can appear in several pairs, so caching avoids re-decoding it.
_EXTRACTOR = KeyframeExtractor()
_frame_cache = {}


def _font(size=12):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _shot_num(shot_id):
    m = re.search(r"shot_(\d+)", shot_id)
    return int(m.group(1)) if m else 0


def _video_frames(video_path):
    """Whole-video decode (320x180 RGB), cached. None if decode fails/missing."""
    if video_path not in _frame_cache:
        if not os.path.exists(video_path):
            print(f"  [warn] video not found: {video_path}")
            _frame_cache[video_path] = None
        else:
            try:
                _frame_cache[video_path] = _EXTRACTOR.get_frames(video_path)
            except Exception as e:
                print(f"  [warn] decode failed {video_path}: {e}")
                _frame_cache[video_path] = None
    return _frame_cache[video_path]


def _parse_splits(splits):
    """fullmatch stores shotsplit as a list-literal string; return a list of ints."""
    if isinstance(splits, str):
        splits = ast.literal_eval(splits)
    return list(splits)


def _num_shots(splits):
    """Shot count, replicating s1_keyframes' leading-0 convention."""
    splits = list(splits)
    if not splits:
        return 0
    if splits[0] != 0:
        splits = [0] + splits
    return len(splits) - 1


def _p50_index(splits, shot_num, pct):
    """Frame index of shot `shot_num` (1-based) at percentile `pct`.

    Mirrors s1_keyframes.extract_keyframes exactly: prepend 0, clamp the shot end
    to total_frames (= last split), require shot_len >= 5 (shorter shots have no
    pNN keyframe), and drop indices past total_frames. Returns None if no p50.
    """
    splits = list(splits)
    if not splits:
        return None
    if splits[0] != 0:
        splits = [0] + splits
    total = splits[-1]
    i = shot_num - 1
    if i < 0 or i + 1 >= len(splits):
        return None
    start = splits[i]
    end = min(splits[i + 1], total)
    shot_len = end - start
    if shot_len < 5:
        return None
    idx = start + int((pct / 100.0) * shot_len)
    if idx >= total:
        return None
    return idx


def _frame_image(video_path, splits, shot_num, pct):
    """PIL image of the pct-percentile frame for shot_num, or None."""
    idx = _p50_index(splits, shot_num, pct)
    if idx is None:
        return None
    frames = _video_frames(video_path)
    if frames is None or idx >= frames.shape[0]:
        return None
    return Image.fromarray(frames[idx])


def _cell(img, label, font):
    """Build one labelled cell (label strip on top, image below)."""
    cell = Image.new("RGB", (CELL_W, LABEL_H + CELL_H), (245, 245, 245))
    draw = ImageDraw.Draw(cell)
    draw.text((3, 3), label, fill=(20, 20, 20), font=font)
    if img is not None:
        im = img.convert("RGB")
        im.thumbnail((CELL_W, CELL_H))
        ox = (CELL_W - im.width) // 2
        oy = LABEL_H + (CELL_H - im.height) // 2
        cell.paste(im, (ox, oy))
    else:
        draw.rectangle([0, LABEL_H, CELL_W - 1, LABEL_H + CELL_H - 1], outline=(180, 180, 180))
        draw.text((CELL_W // 2 - 28, LABEL_H + CELL_H // 2 - 6), "no match", fill=(150, 150, 150), font=font)
    return cell


def render_pair(row, video_root, output_dir, pct=50):
    brand_id = int(row["brand_id"])
    focal = row["focal_ad_id"]      # long
    target = row["target_ad_id"]    # short
    mapping = row["mapping"]
    if isinstance(mapping, str):
        mapping = ast.literal_eval(mapping)

    focal_splits = _parse_splits(row["focal_shotsplit"])
    target_splits = _parse_splits(row["target_shotsplit"])
    focal_video = os.path.join(video_root, row["focal_relative_path"])
    target_video = os.path.join(video_root, row["target_relative_path"])

    # invert: short_shot -> focal(long)_shot  (first focal shot if several map)
    short_to_focal = {}
    for focal_shot, short_shot in mapping.items():
        if short_shot is not None and short_shot not in short_to_focal:
            short_to_focal[short_shot] = focal_shot

    # columns = the short(target) video's shots that actually have a p50 frame
    # (shot_len >= 5), in shot order.
    short_shots = [
        f"shot_{n:02d}"
        for n in range(1, _num_shots(target_splits) + 1)
        if _p50_index(target_splits, n, pct) is not None
    ]
    if not short_shots:
        print(f"  [skip] no p50 shots for target {target} (brand {brand_id})")
        return None

    font = _font(12)
    hfont = _font(14)
    n = len(short_shots)
    W = PAD + n * (CELL_W + PAD)
    H = HEADER_H + PAD + 2 * (LABEL_H + CELL_H + PAD)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    deg = row.get("mapping_degree", "")
    draw.text((PAD, 7),
              f"brand {brand_id}   SHORT(target)={target}  |  LONG(focal)={focal}   "
              f"mapping_degree={deg}", fill=(0, 0, 0), font=hfont)

    y_top = HEADER_H + PAD
    y_bot = y_top + LABEL_H + CELL_H + PAD
    for i, ss in enumerate(short_shots):
        x = PAD + i * (CELL_W + PAD)
        # top: short(target) shot
        top = _cell(_frame_image(target_video, target_splits, _shot_num(ss), pct),
                    f"{target} {ss}", font)
        canvas.paste(top, (x, y_top))
        # bottom: mapped focal (long) shot
        fs = short_to_focal.get(ss)
        bot = _cell(_frame_image(focal_video, focal_splits, _shot_num(fs), pct) if fs else None,
                    f"{focal} {fs}" if fs else f"{focal} —", font)
        canvas.paste(bot, (x, y_bot))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(output_dir) / f"{brand_id}_{focal}__{target}.png"
    canvas.save(out)
    return out


def main():
    import pandas as pd
    p = argparse.ArgumentParser()
    p.add_argument("--fullmatch_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--video_dir", default=config.VIDEO_BASE_DIR,
                   help="Video base dir; relative_path columns are rooted at its parent "
                        "(matches the s1 stages). Defaults to config.VIDEO_BASE_DIR.")
    p.add_argument("--percentile", type=int, default=50, help="keyframe percentile = mid frame (50)")
    p.add_argument("--brand", type=int, default=None, help="optional: only this brand_id")
    args = p.parse_args()

    # relative_path is like "videomedia/<folder>/<ad_id>.mp4", rooted at the
    # parent of the video dir (same convention as s1_keyframes).
    video_root = str(Path(args.video_dir).parent)

    df = pd.read_csv(args.fullmatch_csv)
    if args.brand is not None:
        df = df[df["brand_id"] == args.brand]
    print(f"Rendering {len(df)} pair(s) -> {args.output_dir}")
    for _, row in df.iterrows():
        out = render_pair(row, video_root, args.output_dir, args.percentile)
        if out:
            print(f"  wrote {out}")


if __name__ == "__main__":
    main()
