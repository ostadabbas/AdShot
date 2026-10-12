"""
Build batch_test_videos_fix.csv: a tiny subset of batch1 for full-pipeline testing.

7 brands picked by cross-referencing data/fullmatch823.csv (matched by brand NAME,
since its brand_id space differs from batch1):
  - 4 pair-producing brands  -> verify Stage 3 emits correct pairs
  - 3 no-pair brands          -> verify Stage 3 emits ZERO pairs (no false positives)

Run:
  /projects/adsum-mllm-om/miniconda3/envs/shot/bin/python make_batch_test.py
"""
import pandas as pd

SRC = "/projects/adsum-mllm-om/data2/batch1_videos_fix.csv"
OUT = "/projects/adsum-mllm-om/data2/batch_test_videos_fix.csv"

# brand_id in batch1
PAIR_BRANDS   = [105, 131, 60, 125]   # Coors Light, Visionworks, Lovesac, Capital One
NOPAIR_BRANDS = [12, 15, 18]          # PediaSure, priceline, Hershey's
KEEP = PAIR_BRANDS + NOPAIR_BRANDS

df = pd.read_csv(SRC)
sub = df[df["brand_id"].isin(KEEP)].copy()
sub.to_csv(OUT, index=False)

print(f"wrote {OUT}")
print(f"rows: {len(sub)}  brands: {sub['brand_id'].nunique()}")
print(sub.groupby(["brand_id", "brand_name"])["ad_id"].nunique()
        .rename("videos").reset_index().to_string(index=False))
