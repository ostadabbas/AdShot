"""
Step 4 — embeddings (s1_embedding.Swin3DEmbedder / EmbeddingGenerator).

Gate (heavy: loads Swin3D-B): embeddings generated from the *reference* keyframes
for the test ad must match the stored reference embeddings by cosine similarity
>= COSINE_FLOOR (tolerant: small CPU<->GPU float drift is expected; steps 2-4 saw
mean cosine 0.999998). Feeding reference keyframes isolates this from keyframe
extraction.
"""

import os
import tempfile
import shutil
from _harness import (
    check, require, SkipTest, BRAND_ID, TEST_AD,
    REF_KEYFRAMES_BRAND_DIR, REF_EMBEDDINGS_H5,
    load_h5_embeddings, compare_embeddings, COSINE_FLOOR,
)

META = {"step": 4, "name": "embedding", "heavy": True}


def run():
    require(REF_KEYFRAMES_BRAND_DIR, "reference keyframes")
    require(REF_EMBEDDINGS_H5, "reference embeddings")

    from s1_keyframes import KeyframeExtractor
    from s1_embedding import EmbeddingGenerator, load_embeddings

    # Valid shots (exactly 5 keyframes) for the test ad, from reference keyframes.
    shots = [(sid, fps) for sid, fps in
             KeyframeExtractor().find_valid_shots(REF_KEYFRAMES_BRAND_DIR)
             if sid.startswith(TEST_AD)]
    check(shots, f"no valid reference shots for {TEST_AD}")

    tmp = tempfile.mkdtemp(prefix="emb_test_")
    out_h5 = os.path.join(tmp, "out.h5")
    try:
        EmbeddingGenerator().process_brand(shots, BRAND_ID, out_h5, resume=False)
        produced = load_h5_embeddings(out_h5)
        reference = load_h5_embeddings(REF_EMBEDDINGS_H5)

        # shape sanity
        _e, _b, ids = load_embeddings(out_h5)
        check(all(len(produced[i]) == 1024 for i in produced), "embedding dim != 1024")

        n, mn, mean = compare_embeddings(produced, reference, floor=COSINE_FLOOR, only_prefix=TEST_AD)
        return f"{TEST_AD}: compared {n} shots, min cos={mn:.6f}, mean={mean:.6f} (floor {COSINE_FLOOR})"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print(run())
