"""
Step 9 — batched embeddings.

Gate: Swin3DEmbedder.get_embeddings_batch (one GPU forward over many shots) must
match the per-shot get_embedding (batch-1) by cosine >= COSINE_FLOOR — batching
must not change results. Also sweeps a couple of batch sizes.
"""

from _harness import (
    check, require, cosine, BRAND_ID, TEST_AD,
    REF_KEYFRAMES_BRAND_DIR, COSINE_FLOOR,
)

META = {"step": 9, "name": "embed_batch", "heavy": True}


def run():
    require(REF_KEYFRAMES_BRAND_DIR, "reference keyframes")

    from s1_keyframes import KeyframeExtractor
    from s1_embedding import Swin3DEmbedder

    shots = [(sid, fps) for sid, fps in
             KeyframeExtractor().find_valid_shots(REF_KEYFRAMES_BRAND_DIR)
             if sid.startswith(TEST_AD)]
    check(shots, f"no valid reference shots for {TEST_AD}")

    emb = Swin3DEmbedder()

    # (a) batch-1 vs batched-all equivalence
    single = {sid: emb.get_embedding(fps) for sid, fps in shots}
    batched = emb.get_embeddings_batch([fps for _, fps in shots])
    mn = 1.0
    for (sid, _), vec in zip(shots, batched):
        c = cosine(single[sid], vec)
        mn = min(mn, c)
        check(c >= COSINE_FLOOR, f"shot {sid}: batched vs single cosine {c:.6f} < {COSINE_FLOOR}")

    # sweep a sub-batch size — first 4 shots in one forward must match singles too
    if len(shots) >= 4:
        sub = emb.get_embeddings_batch([fps for _, fps in shots[:4]])
        for (sid, _), vec in zip(shots[:4], sub):
            check(cosine(single[sid], vec) >= COSINE_FLOOR, f"sub-batch drift on {sid}")

    return f"{TEST_AD}: {len(shots)} shots, batched-vs-single min cos={mn:.6f}"


if __name__ == "__main__":
    print(run())
