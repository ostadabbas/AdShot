#!/usr/bin/env python3
"""
split_test_eval.py — split the 124-row held-out test set into an in-loop EVAL split
(used for per-epoch validation during training) and a final TEST split (evaluated once
at the end of training).

The csv<->jsonl join is POSITIONAL (line i of jsonl aligns with row i of csv), so both
files are permuted with the SAME seeded permutation and sliced at the SAME index. The
original data/test.{csv,jsonl} are left untouched (v1 comparisons stay reproducible).

  python data/split_test_eval.py --frac_eval 0.5 --seed 0
Outputs: data/eval.{csv,jsonl}, data/test_final.{csv,jsonl}
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/test.csv")
    ap.add_argument("--jsonl", default="data/test.jsonl")
    ap.add_argument("--frac_eval", type=float, default=0.5,
                    help="Fraction of rows that go to the in-loop eval split.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    with open(args.jsonl) as f:
        jl = [line.rstrip("\n") for line in f]
    assert len(df) == len(jl), f"csv {len(df)} vs jsonl {len(jl)} length mismatch"
    n = len(df)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_eval = int(round(n * args.frac_eval))
    eval_idx = sorted(perm[:n_eval].tolist())
    test_idx = sorted(perm[n_eval:].tolist())
    assert set(eval_idx).isdisjoint(test_idx)
    assert len(eval_idx) + len(test_idx) == n

    out = Path(args.out_dir)

    def write_split(name, idx):
        sub_df = df.iloc[idx].reset_index(drop=True)
        sub_df.to_csv(out / f"{name}.csv", index=False)
        with open(out / f"{name}.jsonl", "w") as f:
            for i in idx:
                f.write(jl[i] + "\n")
        # verify positional alignment: focal_ad_id must appear in the jsonl id at each row
        for r in range(len(sub_df)):
            fa = str(sub_df.loc[r, "focal_ad_id"])
            jid = json.loads(jl[idx[r]])["id"]
            assert fa in jid, f"{name} row {r}: focal_ad_id {fa} not in id {jid}"
        print(f"{name}: {len(idx)} rows -> {out/f'{name}.csv'}, {out/f'{name}.jsonl'} (aligned OK)")

    write_split("eval", eval_idx)
    write_split("test_final", test_idx)
    print(f"DONE: {n} -> eval {len(eval_idx)} + test_final {len(test_idx)} (seed={args.seed})")


if __name__ == "__main__":
    main()
