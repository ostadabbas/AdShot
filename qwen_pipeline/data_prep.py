#!/usr/bin/env python3
"""
data_prep.py — combine the locked-0.8 pair-extraction outputs into a single
training/eval dataset for the Qwen3-VL shot-selection task.

Pipeline:
  1. Concatenate data2/batch{1..5}/fullmatch.csv  -> data/fullmatch_all.csv
     (adds a `batch` column for traceability).
  2. For each pair compute the TARGET = the focal shots that compose the short
     ad, i.e. the focal shots whose `mapping` value is not None. Unmatched short
     shots simply do not exist in the target ([1,3,4,X,X,5,6] -> [1,3,4,5,6]).
     The focal INPUT is never modified.
  3. Compute focal shot durations WITHOUT decoding video (same math as
     evaluations/build_message.py:67) so total_length_sec is exact.
  4. Random train/test split (by row, configurable) -> train/test .csv + .jsonl.

The JSONL is a task record (ShareGPT-style conversations) carrying everything
needed to drive SFT or analysis. Per-shot frame image paths are NOT written here
(that needs ffmpeg extraction) — run with --export_frames once frame_cache.py is
in place, or let dataset.py build messages lazily at train/infer time.

Run:
  /projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python data_prep.py
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

# Reuse the eval utilities (cross-check the target set).
PROJECT_ROOT = Path("/projects/adsum-mllm-om")
EVAL_DIR = PROJECT_ROOT / "mllm" / "evaluations"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from utils import extract_shots_from_mapping, create_one_hot  # noqa: E402

DATA2 = PROJECT_ROOT / "data2"
DEFAULT_BATCHES = [1, 2, 3, 4, 5]
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data"


def _as_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else list(x)


def shot_ranges(shotsplit):
    """(start, end) inclusive frame ranges per shot — mirrors build_message."""
    ranges, start = [], 0
    for end in shotsplit:
        ranges.append((start, end))
        start = end + 1
    return ranges


def shot_durations(shotsplit, ad_length):
    """Per-focal-shot duration in seconds, no video decode."""
    splits = _as_list(shotsplit)
    ad_length = float(ad_length)
    fps = (splits[-1] + 1) / ad_length
    return [(e - s + 1) / fps for s, e in shot_ranges(splits)]


def selected_focal_shots(mapping_str):
    """Focal shot numbers (1-based, ascending) that map to a short shot.

    This is the 'drop the X' collapse expressed as a set: a focal shot is in the
    target iff some short shot maps to it. Equivalent to the set returned by
    utils.extract_shots_from_mapping(...)[0].
    """
    m = ast.literal_eval(mapping_str)
    return sorted(int(k.split("_")[1]) for k, v in m.items() if v is not None)


def build_record(row, fps_label):
    """One ShareGPT-style task record (no frame paths yet)."""
    shotsplit = _as_list(row["focal_shotsplit"])
    durs = shot_durations(shotsplit, row["focal_ad_length"])
    n_shots = len(durs)
    selected = selected_focal_shots(row["mapping"])
    total_len = round(sum(durs[i - 1] for i in selected), 2)

    gpt_answer = {
        "selected_shots": [f"shot{i}" for i in selected],
        "total_length_sec": total_len,
    }
    return {
        "id": f"{int(row['brand_id'])}_{row['focal_ad_id']}__{row['target_ad_id']}",
        "video": row["focal_relative_path"],            # focal (long) video
        "batch": int(row["batch"]),
        "focal_shotsplit": shotsplit,
        "focal_ad_length": float(row["focal_ad_length"]),
        "n_shots": n_shots,
        "shot_durations": {f"shot{i}": round(d, 2) for i, d in enumerate(durs, 1)},
        "selected_shots": gpt_answer["selected_shots"],
        "total_length_sec": total_len,
        "unmatched_shots": int(row["unmatched_shots"]),
        # The gpt target answer. Full system/human turns + <image> frame refs are
        # assembled at train/infer time (dataset.py) or by --export_frames, which
        # reuse build_message so the prompt stays identical to eval.
        "answer": gpt_answer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--split_by", choices=["row", "brand"], default="row")
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps_label", type=int, default=5,
                    help="Frames-per-shot label baked into JSONL metadata (prompt is built at train/infer time).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) combine
    frames = []
    for b in args.batches:
        f = DATA2 / f"batch{b}" / "fullmatch.csv"
        if not f.exists():
            raise FileNotFoundError(f)
        df = pd.read_csv(f)
        df["batch"] = b
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full_path = out_dir / "fullmatch_all.csv"
    full.to_csv(full_path, index=False)
    print(f"[combine] {len(full)} pairs from batches {args.batches} -> {full_path}")

    # 2) cross-check target == extract_shots_from_mapping set (sample)
    mism = 0
    for _, r in full.iterrows():
        sel = set(selected_focal_shots(r["mapping"]))
        ref = set(extract_shots_from_mapping(r["mapping"])[0])
        if sel != ref:
            mism += 1
    print(f"[check] target-set vs extract_shots_from_mapping mismatches: {mism} (expect 0)")
    assert mism == 0, "target collapse disagrees with utils.extract_shots_from_mapping"

    # 3) split
    rng_full = full.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.split_by == "brand":
        brands = rng_full["brand_id"].drop_duplicates().tolist()
        n_test = max(1, int(round(len(brands) * args.test_frac)))
        test_brands = set(brands[:n_test])
        test = rng_full[rng_full["brand_id"].isin(test_brands)]
        train = rng_full[~rng_full["brand_id"].isin(test_brands)]
    else:  # row
        n_test = int(round(len(rng_full) * args.test_frac))
        test = rng_full.iloc[:n_test]
        train = rng_full.iloc[n_test:]

    assert len(train) + len(test) == len(full)
    overlap = set(train.index) & set(test.index)
    assert not overlap, "train/test row overlap"

    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(f"[split] by {args.split_by}: train={len(train)} test={len(test)} "
          f"(test_frac={args.test_frac}, seed={args.seed})")

    # 4) JSONL task records
    for name, part in [("train", train), ("test", test)]:
        jpath = out_dir / f"{name}.jsonl"
        with jpath.open("w", encoding="utf-8") as fh:
            for _, r in part.iterrows():
                fh.write(json.dumps(build_record(r, args.fps_label), ensure_ascii=False) + "\n")
        print(f"[jsonl] wrote {len(part)} records -> {jpath}")

    # quick stats
    sel_counts = full["mapping"].apply(lambda m: len(selected_focal_shots(m)))
    print(f"[stats] selected focal shots/pair: mean={sel_counts.mean():.2f} "
          f"min={sel_counts.min()} max={sel_counts.max()}")
    print(f"[stats] pairs with unmatched_shots>0: {(full['unmatched_shots']>0).sum()}/{len(full)}")


if __name__ == "__main__":
    main()
