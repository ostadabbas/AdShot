"""
Step 3 — keyframe extraction (s1_keyframes.KeyframeExtractor).

Gate (heavy: decodes the video via ffmpeg): extracting keyframes for the test ad
into a scratch dir must reproduce, for that ad's filename prefix, the exact set
of keyframe filenames found in the reference keyframes dir, and a sampled subset
must be byte-identical (md5). Uses the reference shotsplit as input so this tests
keyframe extraction in isolation from shot detection.
"""

import os
import tempfile
import shutil
from _harness import (
    check, require, SkipTest, BRAND_ID, BRAND_NAME, TEST_AD,
    REF_SHOTSPLITS_JSON, REF_KEYFRAMES_BRAND_DIR, load_shotsplits,
    keyframe_names, md5,
)

META = {"step": 3, "name": "keyframes", "heavy": True}

VIDEO = f"/projects/adsum-mllm-om/videomedia/{BRAND_NAME}/{TEST_AD}.mp4"


def run():
    require(VIDEO, "test video")
    require(REF_SHOTSPLITS_JSON, "reference shotsplits")
    require(REF_KEYFRAMES_BRAND_DIR, "reference keyframes")

    ref_splits = load_shotsplits(REF_SHOTSPLITS_JSON)
    check(TEST_AD in ref_splits, f"{TEST_AD} not in reference shotsplits")
    shotsplit = ref_splits[TEST_AD]

    ref_names = keyframe_names(REF_KEYFRAMES_BRAND_DIR, prefix=TEST_AD)
    check(ref_names, f"no reference keyframes for prefix {TEST_AD}")

    from s1_keyframes import KeyframeExtractor

    tmp = tempfile.mkdtemp(prefix="kf_test_")
    try:
        KeyframeExtractor().extract_keyframes(VIDEO, shotsplit, TEST_AD, tmp, set())
        got_names = keyframe_names(tmp, prefix=TEST_AD)

        check(got_names == ref_names,
              "keyframe filename set mismatch:\n"
              f"  missing={sorted(ref_names - got_names)[:5]}\n"
              f"  extra={sorted(got_names - ref_names)[:5]}")

        # Byte-compare up to 5 sampled frames.
        sample = sorted(got_names)[:5]
        for fn in sample:
            check(md5(os.path.join(tmp, fn)) == md5(os.path.join(REF_KEYFRAMES_BRAND_DIR, fn)),
                  f"keyframe {fn} not byte-identical to reference")
        return f"{TEST_AD}: {len(got_names)} keyframes, filename set + {len(sample)} md5s match"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print(run())
