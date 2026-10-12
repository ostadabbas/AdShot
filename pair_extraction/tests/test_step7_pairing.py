"""
Step 7 — pairing (s3_pairing.run_pairing), the orphaned stage now wired.

Gate (no GPU): run pairing against the FULL reference data/0.1 artifacts
(shotsplits + clusters) and the batch1..5 *_videos_wen.csv inputs that produced
the committed data/fullmatch.csv, writing to a scratch path. The matching
algorithm is deterministic, so the regenerated fullmatch must match the
reference exactly:
  - identical column set
  - identical row count
  - identical focal_ad_id set (rows are de-duped on focal_ad_id)

Contract for Step 7 implementation — s3_pairing.run_pairing must accept these
keyword overrides (each defaulting to config when None):
    run_pairing(shotsplits_dir=None, clusters_dir=None,
                data_csvs=None, out_path=None) -> str (path written)

Skips until s3_pairing.py is added to pair_extraction (Step 7).
"""

import os
import tempfile
import shutil
from _harness import (
    check, require, require_module, SkipTest,
    REF_SHOTSPLITS_DIR, REF_CLUSTERS_DIR, REF_FULLMATCH,
)

META = {"step": 7, "name": "pairing", "heavy": True}

DATA_ROOT = "/projects/adsum-mllm-om/data2"
WEN_CSVS = [os.path.join(DATA_ROOT, f"batch{i}_videos_wen.csv") for i in range(1, 6)]


def run():
    require_module("s3_pairing", "Step 7: pairing")
    require(REF_SHOTSPLITS_DIR, "reference shotsplits dir")
    require(REF_CLUSTERS_DIR, "reference clusters dir")
    require(REF_FULLMATCH, "reference fullmatch.csv")
    for c in WEN_CSVS:
        require(c, "batch wen csv")

    import pandas as pd
    import s3_pairing

    check(hasattr(s3_pairing, "run_pairing"), "s3_pairing.run_pairing missing")

    tmp = tempfile.mkdtemp(prefix="pair_test_")
    out_path = os.path.join(tmp, "fullmatch.csv")
    try:
        s3_pairing.run_pairing(
            shotsplits_dir=REF_SHOTSPLITS_DIR,
            clusters_dir=REF_CLUSTERS_DIR,
            data_csvs=WEN_CSVS,
            out_path=out_path,
        )
        check(os.path.exists(out_path), f"fullmatch not written: {out_path}")

        got = pd.read_csv(out_path)
        ref = pd.read_csv(REF_FULLMATCH)

        check(set(got.columns) == set(ref.columns),
              f"column mismatch: missing={set(ref.columns)-set(got.columns)} "
              f"extra={set(got.columns)-set(ref.columns)}")
        check(len(got) == len(ref), f"row count mismatch: got={len(got)} exp={len(ref)}")
        check(set(got["focal_ad_id"]) == set(ref["focal_ad_id"]),
              "focal_ad_id set differs from reference")

        return f"fullmatch reproduced: {len(got)} rows, {len(got.columns)} cols, focal set matches"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print(run())
