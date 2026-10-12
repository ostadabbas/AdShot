"""
Step 5 — Stage 1 driver (s1_1_process_brands_main.BrandVideoProcessor).

End-to-end integration test: chains shot detection -> keyframes -> embeddings for
the test brand into a scratch output dir, then checks the chained outputs against
the reference for the one ad whose video exists locally (XEEvoOwsqEY):
  - shotsplit: exact integer-list match
  - keyframes: exact filename-set match (for the ad prefix)
  - embeddings: cosine >= COSINE_FLOOR (tolerant; CPU/GPU drift)
Also confirms the status file (processing_status.json) is written.

Skips until s1_1_process_brands_main.py is wired into pair_extraction (Step 5),
or if the test video is missing.
"""

import os
import tempfile
import shutil
from _harness import (
    check, require, require_module, SkipTest,
    BRAND_ID, BRAND_NAME, TEST_AD,
    REF_SHOTSPLITS_JSON, REF_KEYFRAMES_BRAND_DIR, REF_EMBEDDINGS_H5,
    load_shotsplits, keyframe_names, load_h5_embeddings, compare_embeddings,
    COSINE_FLOOR,
)

META = {"step": 5, "name": "stage1_driver", "heavy": True}

VIDEO = f"/projects/adsum-mllm-om/videomedia/{BRAND_NAME}/{TEST_AD}.mp4"


def run():
    require_module("s1_1_process_brands_main", "Step 5: wire Stage 1 driver")
    require(VIDEO, "test video")
    require(REF_SHOTSPLITS_JSON, "reference shotsplits")
    require(REF_EMBEDDINGS_H5, "reference embeddings")

    import pandas as pd
    import s1_1_process_brands_main as driver

    # Minimal one-row data frame for the test brand (only the ad with a local video).
    data_df = pd.DataFrame([{
        "brand_id": BRAND_ID,
        "brand_name": BRAND_NAME,
        "ad_id": TEST_AD,
        "ad_length": 30,
    }])

    out_root = tempfile.mkdtemp(prefix="stage1_test_")
    try:
        proc = driver.BrandVideoProcessor(
            data_df=data_df,
            video_base_dir="/projects/adsum-mllm-om/videomedia",
            output_base_dir=out_root,
            num_workers=1,
        )
        ok = proc.process_single_brand(BRAND_NAME)
        check(ok, "driver.process_single_brand returned failure")

        # status file written
        check(os.path.exists(os.path.join(out_root, "processing_status.json")),
              "processing_status.json not written")

        # --- shotsplit (exact) ---
        got_shots = load_shotsplits(
            os.path.join(out_root, "shotsplits", f"{BRAND_ID}_{BRAND_NAME}_shotsplits.json"))
        ref_shots = load_shotsplits(REF_SHOTSPLITS_JSON)
        check(list(got_shots[TEST_AD]) == list(ref_shots[TEST_AD]),
              f"shotsplit mismatch: got={got_shots.get(TEST_AD)} exp={ref_shots[TEST_AD]}")

        # --- keyframes (exact filename set for this ad) ---
        got_kf = keyframe_names(os.path.join(out_root, "keyframes", str(BRAND_ID)), prefix=TEST_AD)
        ref_kf = keyframe_names(REF_KEYFRAMES_BRAND_DIR, prefix=TEST_AD)
        check(got_kf == ref_kf,
              f"keyframe set mismatch: missing={sorted(ref_kf-got_kf)[:3]} extra={sorted(got_kf-ref_kf)[:3]}")

        # --- embeddings (tolerant cosine) ---
        produced = load_h5_embeddings(
            os.path.join(out_root, "embeddings", f"{BRAND_ID}_{BRAND_NAME}_embeddings.h5"))
        reference = load_h5_embeddings(REF_EMBEDDINGS_H5)
        n, mn, mean = compare_embeddings(produced, reference, only_prefix=TEST_AD)

        return (f"{TEST_AD}: shots+{len(got_kf)} keyframes exact; "
                f"{n} embeddings cos min={mn:.6f} mean={mean:.6f}")
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


if __name__ == "__main__":
    print(run())
