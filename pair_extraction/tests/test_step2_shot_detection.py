"""
Step 2 — shot detection (s1_shot_detection.ShotDetector).

Gate (heavy: loads AutoShot + decodes the video): detected shot boundaries for
the test ad must equal the stored reference shotsplit exactly. Shot detection is
deterministic, so this is an exact integer-list comparison.
"""

import os
from _harness import (
    check, require, SkipTest, BRAND_NAME, TEST_AD,
    REF_SHOTSPLITS_JSON, load_shotsplits,
)

META = {"step": 2, "name": "shot_detection", "heavy": True}

VIDEO = f"/projects/adsum-mllm-om/videomedia/{BRAND_NAME}/{TEST_AD}.mp4"


def run():
    require(VIDEO, "test video")
    require(REF_SHOTSPLITS_JSON, "reference shotsplits")
    ref_splits = load_shotsplits(REF_SHOTSPLITS_JSON)
    check(TEST_AD in ref_splits, f"{TEST_AD} not in reference shotsplits")
    expected = ref_splits[TEST_AD]

    import config
    from s1_shot_detection import ShotDetector

    detector = ShotDetector(
        model_path=config.AUTOSHOT_MODEL_PATH,
        threshold=config.SHOT_THRESHOLD,
    )
    shot_splits, _preds = detector.detect_shots(VIDEO)

    check(list(shot_splits) == list(expected),
          f"shotsplit mismatch for {TEST_AD}:\n  got={shot_splits}\n  exp={expected}")
    return f"{TEST_AD}: {len(shot_splits)} boundaries, exact match to reference"


if __name__ == "__main__":
    print(run())
