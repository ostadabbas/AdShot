"""
Step 0: brand_id normalization.

The raw download-pipeline CSVs (batchN_videos.csv) carry brand_id like 'b1306',
while the pair-extraction stages expect the bare integer form ('1306'). This
step strips the leading 'b' and writes batchN_videos_fix.csv.

Some hand-curated brands were never assigned a real numeric id (the source
industry CSVs used placeholders like 'o1'..'o4', which end up empty/non-numeric
in the videos CSV). Those would crash later stages, e.g. `int(row["brand_id"])`
in s1bc_fused.py. This step backfills a fresh, unique brand_id per such brand,
keyed by brand_name, using a counter that starts above all existing ids (floor
ID_FLOOR=20001) so it cannot collide with real ids.

Verified to reproduce the existing *_fix.csv files exactly for batches with no
missing ids (full-frame identical on batch7 and batch9). This is the
productionized version of the old ad-hoc fix_brand_id.py.

Usage:
    python s0_fix_brand_id.py --input  /path/to/batch7_videos.csv
    python s0_fix_brand_id.py --input  batch7_videos.csv --output batch7_videos_fix.csv
"""

import argparse
import os
import pandas as pd


# Floor for backfilled brand_ids. Existing real ids are well below this; the
# '20000+' range matches the convention already used for the 'o1'..'o4' brands
# in other batches (Gillette Italia=20002, Scope Mouthwash=20003, ...).
ID_FLOOR = 20001


def sanitize_brand_name(df: pd.DataFrame) -> pd.DataFrame:
    """Replace path separators in brand_name with '_'.

    brand_name is used to build per-brand output filenames like
    '{brand_id}_{brand_name}_shotsplits.json' / '..._embeddings.h5'. A '/' (e.g.
    'RE/MAX') is read as a directory separator and crashes the open() with
    FileNotFoundError. Video files are resolved via the authoritative
    `relative_path` column, so renaming the label here is safe. Clean names are
    unchanged, keeping output byte-identical for batches without separators.
    """
    out = df.copy()
    sanitized = out["brand_name"].astype(str).str.replace(r"[/\\]", "_", regex=True)
    changed = sanitized != out["brand_name"].astype(str)
    if changed.any():
        for old in out.loc[changed, "brand_name"].astype(str).unique():
            new = old.replace("/", "_").replace("\\", "_")
            print(f"[step0] sanitized brand_name '{old}' -> '{new}' (path separator)")
        out["brand_name"] = sanitized
    return out


def fix_brand_id(df: pd.DataFrame, id_floor: int = ID_FLOOR) -> pd.DataFrame:
    """Strip a leading 'b' from brand_id (b1306 -> 1306), then backfill a fresh
    unique id for any brand whose id is empty/non-numeric.

    For clean batches (every id numeric after stripping) the output is identical
    to the original strip-only behaviour.
    """
    out = df.copy()
    stripped = out["brand_id"].astype(str).str.replace(r"^b", "", regex=True)
    numeric = pd.to_numeric(stripped, errors="coerce")
    invalid = numeric.isna()

    if not invalid.any():
        out["brand_id"] = stripped
        return out

    # Counter starts above every valid existing id (and never below the floor).
    existing_max = numeric.max()
    next_id = id_floor if pd.isna(existing_max) else max(id_floor, int(existing_max) + 1)

    def _group_key(idx):
        bn = out.at[idx, "brand_name"]
        if pd.isna(bn) or str(bn).strip() == "":
            return f"__row_{idx}__"  # no brand_name: give the row its own id
        return str(bn)

    new_ids = stripped.copy()
    assigned = {}  # group_key -> new id
    for idx in out.index[invalid]:
        key = _group_key(idx)
        if key not in assigned:
            assigned[key] = next_id
            next_id += 1
        new_ids.at[idx] = str(assigned[key])

    out["brand_id"] = new_ids
    for key, nid in assigned.items():
        n = int((new_ids[invalid] == str(nid)).sum())
        print(f"[step0] backfilled brand_id {nid} for '{key}' ({n} rows, was empty/non-numeric)")
    return out


def dedup_ad_id(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate ad_id rows (keep first).

    Some batches list the same video more than once. Every observed duplicate is
    the same brand_name and same relative_path, so keeping the first row loses
    nothing the pipeline uses. Left in, a duplicate ad_id becomes two embedding
    tasks in s1bc_fused (no dedup there), embedding the same shots twice into the
    brand H5 and double-counting in clustering/pairing. Batches with no duplicate
    ad_ids are returned unchanged (byte-identical output preserved).
    """
    if "ad_id" not in df.columns:
        return df
    dup = df["ad_id"].duplicated(keep="first")
    n = int(dup.sum())
    if n:
        print(f"[step0] dropped {n} duplicate ad_id rows ({df['ad_id'][dup].nunique()} distinct ad_ids)")
        return df[~dup].reset_index(drop=True)
    return df


def default_output_path(input_csv: str) -> str:
    """batchN_videos.csv -> batchN_videos_fix.csv (in the same directory)."""
    root, ext = os.path.splitext(input_csv)
    if root.endswith("_videos"):
        return f"{root}_fix{ext}"
    return f"{root}_fix{ext}"


def run_fix(input_csv: str, output_csv: str | None = None) -> str:
    """Read a raw batch CSV, normalize brand_id, write the *_fix.csv. Returns output path."""
    if output_csv is None:
        output_csv = default_output_path(input_csv)
    df = pd.read_csv(input_csv)
    out = fix_brand_id(df)
    out = sanitize_brand_name(out)
    out = dedup_ad_id(out)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"[step0] {input_csv} -> {output_csv}  ({len(out)} rows)")
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Step 0: normalize brand_id by stripping a leading 'b' (b1306 -> 1306)."
    )
    parser.add_argument("--input", required=True, help="Raw batchN_videos.csv")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <input with _videos -> _videos_fix>.csv)",
    )
    args = parser.parse_args()
    run_fix(args.input, args.output)


if __name__ == "__main__":
    main()
