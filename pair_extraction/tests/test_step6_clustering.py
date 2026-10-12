"""
Step 6 — clustering (s2_clustering.BrandClusteringProcessor).

Gate: cluster the test brand using the *reference* embeddings + keyframes as
input, then compare the produced clusters JSON to the reference clusters JSON.
Because the inputs are the reference embeddings (not regenerated, so no float
drift), DBSCAN is fully deterministic and the result should reproduce the
reference:
  - n_shots: exact
  - n_clusters: exact
  - cluster membership: adjusted Rand index >= ARI_FLOOR (expected 1.0)

Skips until s2_clustering.py is added to pair_extraction (Step 6).
"""

import os
import json
import tempfile
import shutil
from _harness import (
    check, require, require_module, SkipTest,
    BRAND_ID, BRAND_NAME,
    REF_EMBEDDINGS_DIR, REF_KEYFRAMES_DIR, REF_CLUSTERS_JSON,
    cluster_membership_labels, ARI_FLOOR,
)

META = {"step": 6, "name": "clustering", "heavy": False}


def _load_clusters_obj(path):
    """Reference/produced clusters file is a single JSON object (or 1-row JSONL)."""
    with open(path) as f:
        txt = f.read().strip()
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        obj = json.loads(txt.splitlines()[0])
    # the cluster mapping lives under "clusters" (raw file) — tolerate "cluster"
    clusters = obj.get("clusters", obj.get("cluster"))
    return obj, clusters


def run():
    require_module("s2_clustering", "Step 6: clustering")
    require(REF_EMBEDDINGS_DIR, "reference embeddings dir")
    require(REF_CLUSTERS_JSON, "reference clusters")

    from sklearn.metrics import adjusted_rand_score
    import s2_clustering as s2

    tmp = tempfile.mkdtemp(prefix="clu_test_")
    try:
        proc = s2.BrandClusteringProcessor(
            embeddings_dir=REF_EMBEDDINGS_DIR,
            keyframes_dir=REF_KEYFRAMES_DIR,
            output_dir=tmp,
        )
        proc.process_brand_clustering(BRAND_ID, BRAND_NAME)

        out_path = os.path.join(tmp, f"{BRAND_ID}_{BRAND_NAME}_clusters.json")
        check(os.path.exists(out_path), f"clusters file not written: {out_path}")

        got_obj, got_clusters = _load_clusters_obj(out_path)
        ref_obj, ref_clusters = _load_clusters_obj(REF_CLUSTERS_JSON)

        check(got_obj["n_shots"] == ref_obj["n_shots"],
              f"n_shots mismatch: got={got_obj['n_shots']} exp={ref_obj['n_shots']}")
        check(got_obj["n_clusters"] == ref_obj["n_clusters"],
              f"n_clusters mismatch: got={got_obj['n_clusters']} exp={ref_obj['n_clusters']}")

        # Compare membership over the shared shot universe.
        universe = set()
        for c in (got_clusters, ref_clusters):
            for shots in c.values():
                universe.update(shots)
        _, got_labels = cluster_membership_labels(got_clusters, universe)
        _, ref_labels = cluster_membership_labels(ref_clusters, universe)
        ari = adjusted_rand_score(ref_labels, got_labels)
        check(ari >= ARI_FLOOR, f"cluster ARI {ari:.4f} < floor {ARI_FLOOR}")

        return (f"n_shots={got_obj['n_shots']} n_clusters={got_obj['n_clusters']} "
                f"ARI={ari:.4f} vs reference")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print(run())
