"""
Step 0 — brand_id normalization (s0_fix_brand_id.py).

Gate: regenerating a *_fix.csv from a raw *_videos.csv must reproduce the
committed *_fix.csv exactly (the only known diff is CRLF vs LF line endings,
which we normalize away before comparing).
"""

import os
from _harness import check, require, SkipTest

META = {"step": 0, "name": "fix_brand_id", "heavy": False}

DATA_ROOT = "/projects/adsum-mllm-om/data2"
# Batches that ship both a raw _videos.csv and a committed _fix.csv to diff against.
CANDIDATES = ["batch7", "batch9"]


def _norm(path):
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


def run():
    import s0_fix_brand_id as s0

    checked = []
    for b in CANDIDATES:
        raw = os.path.join(DATA_ROOT, f"{b}_videos.csv")
        ref = os.path.join(DATA_ROOT, f"{b}_videos_fix.csv")
        if not (os.path.exists(raw) and os.path.exists(ref)):
            continue
        out = os.path.join(DATA_ROOT, f"{b}_videos_fix__test.csv")
        try:
            s0.run_fix(raw, out)
            check(os.path.exists(out), f"{b}: no output written")
            check(_norm(out) == _norm(ref),
                  f"{b}: regenerated fix.csv differs from committed (beyond CRLF)")
            # also assert no brand_id retains a leading 'b'
            import pandas as pd
            bad = pd.read_csv(out)["brand_id"].astype(str).str.match(r"^b").sum()
            check(bad == 0, f"{b}: {bad} brand_ids still start with 'b'")
            checked.append(b)
        finally:
            if os.path.exists(out):
                os.remove(out)

    if not checked:
        raise SkipTest(f"no raw+fix batch pair found under {DATA_ROOT}")
    return f"reproduced _fix.csv exactly for: {', '.join(checked)}"


if __name__ == "__main__":
    print(run())
