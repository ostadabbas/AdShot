"""
Stage 3 — pairing.

Productionized from the notebook `pair_extraction/2.0-shots-overlapping.py`. The
matching algorithm is kept byte-for-byte; only the hardcoded `data/0.1/...` reads
and the `batch1..5_videos_wen.csv` concat are replaced with parameters that
default to `config` when None.

Flow:
  1. Load shotsplits (JSONL per brand) + clusters (one JSON object per brand).
  2. Merge with the ad metadata CSV(s) to attach ad_length / relative_path.
  3. For each focal (short) ad, map its shots to longer ads in the same brand via
     shared clusters (reverse long->short mapping).
  4. Keep best matches, compute mapping_degree / shot counts.
  5. Filter: mapping_degree >= 0.9 & focal_shot_count >= 4 & target_shot_count >= 2,
     de-dup on focal_ad_id, write fullmatch.csv.

CLI:
    python s3_pairing.py [--shotsplits_dir ...] [--clusters_dir ...]
                         [--data_csv a.csv --data_csv b.csv ...] [--out_path ...]
"""

import argparse
import ast
import os
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

import config

tqdm.pandas()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_shots(shotsplits_dir):
    """Concat all *_shotsplits.json (JSONL), parse shotsplit, de-dup on ad_id."""
    dfs = []
    for fname in os.listdir(shotsplits_dir):
        if "fail" in fname:
            continue
        path = os.path.join(shotsplits_dir, fname)
        try:
            dfs.append(pd.read_json(path, lines=True))
        except Exception:
            print(f"[pairing] could not read shotsplits: {path}")
    shots = pd.concat(dfs, ignore_index=True)
    shots["shotsplit"] = shots["shotsplit"].apply(lambda x: ast.literal_eval(x))
    shots = shots.drop_duplicates(subset="ad_id", keep="first").reset_index(drop=True)
    return shots


def load_clusters(clusters_dir):
    """Concat all *_clusters.json (one JSON object each) into a clusters frame."""
    dfs = []
    for fname in os.listdir(clusters_dir):
        if "fail" in fname:
            continue
        path = os.path.join(clusters_dir, fname)
        try:
            dfs.append(pd.read_json(path, lines=True))
        except Exception as e:
            print(f"[pairing] could not read clusters {path}: {e}")
    clusters = pd.concat(dfs, ignore_index=True)
    clusters.columns = ["brand_id", "brand_name", "n_shots", "best_eps", "best_score",
                        "n_clusters", "cluster"]
    return clusters


def load_data(data_csvs):
    """Concat the ad-metadata CSV(s) (ad_id, ad_length, relative_path, brand_id)."""
    return pd.concat([pd.read_csv(c) for c in data_csvs], ignore_index=True)


# ---------------------------------------------------------------------------
# Matching algorithm (kept identical to 2.0-shots-overlapping.py)
# ---------------------------------------------------------------------------
def standardize_ad_length(length):
    """Standardize ad lengths to 15s or 30s categories; None otherwise."""
    if pd.isna(length):
        return None
    elif length in [14, 15, 16]:
        return 15
    elif length in [29, 30, 31]:
        return 30
    else:
        return None


def process_row_reverse(row, campaigns_dict, cluster_dicts_by_pair):
    long_ad = row["ad_id"]
    pair_id = row["brand_id"]
    length = row["ad_length"]
    cluster_dict = cluster_dicts_by_pair[pair_id]

    shot_to_cluster = {
        shot: cluster_id
        for cluster_id, shots in cluster_dict.items()
        for shot in shots
    }

    long_shots_full = [s for s in shot_to_cluster if s.startswith(long_ad + "_shot_")]
    long_shot_ids = [f'shot_{s.split("_shot_")[-1]}' for s in long_shots_full]

    shorter_ads = [
        ad for ad in campaigns_dict[pair_id]
        if ad["length"] < length and ad["ad_id"] != long_ad
    ]
    if not shorter_ads:
        return {}

    output = {}
    for short_ad in shorter_ads:
        short_ad_id = short_ad["ad_id"]
        shot_mapping = {}
        for long_full in long_shots_full:
            long_shot_id = f'shot_{long_full.split("_shot_")[-1]}'
            cluster = shot_to_cluster.get(long_full)
            target_match = next(
                (s for s in cluster_dict[cluster]
                 if s.startswith(short_ad_id + "_shot_")),
                None
            )
            if target_match:
                matched_shot_id = f'shot_{target_match.split("_shot_")[-1]}'
                shot_mapping[long_shot_id] = matched_shot_id
            else:
                shot_mapping[long_shot_id] = None
        if any(v is not None for v in shot_mapping.values()):
            output[short_ad_id] = shot_mapping
    return output


def sort_shot_keys(d):
    sorted_d = {}
    for long_ad, mapping in d.items():
        sorted_mapping = dict(
            sorted(
                mapping.items(),
                key=lambda x: int(re.search(r"shot_(\d+)", x[0]).group(1))
            )
        )
        sorted_d[long_ad] = sorted_mapping
    return sorted_d


def find_best_matching_ads(focal_to_targets):
    max_overlap = -1
    best_ads = []
    for long_ad, mapping in focal_to_targets.items():
        unique_targets = set(v for v in mapping.values() if v is not None)
        overlap = len(unique_targets)
        if overlap > max_overlap:
            max_overlap = overlap
            best_ads = [long_ad]
        elif overlap == max_overlap:
            best_ads.append(long_ad)
    return best_ads if max_overlap > 0 else []


def unfold_best_matches(df):
    ad_lookup = {
        row["ad_id"]: {"shotsplit": row["shotsplit"], "ad_length": row["ad_length"]}
        for _, row in df.iterrows()
    }
    expanded_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        focal_id = row["ad_id"]
        brand_id = row["brand_id"]
        ad_length = row["ad_length"]
        shotsplit = row["shotsplit"]
        focal_to_targets = row["focal_to_targets"]
        best_ads = row["best_matching_ads"]
        for long_ad in best_ads:
            mapping = focal_to_targets.get(long_ad, {})
            long_info = ad_lookup.get(long_ad, {"shotsplit": None, "ad_length": None})
            expanded_rows.append({
                "brand_id": brand_id,
                "focal_ad_id": focal_id,
                "target_ad_id": long_ad,
                "mapping": mapping,
                "focal_ad_length": ad_length,
                "focal_shotsplit": shotsplit,
                "target_ad_length": long_info["ad_length"],
                "target_shotsplit": long_info["shotsplit"],
            })
    return pd.DataFrame(expanded_rows)


def count_long_shots(shot_split):
    shot_lengths = [shot_split[0]] + [
        end - start for start, end in zip(shot_split[:-1], shot_split[1:])
    ]
    return sum(1 for l in shot_lengths if l >= 5)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pairing(shotsplits_dir=None, clusters_dir=None, data_csvs=None, out_path=None,
                min_focal_shot_count=4, min_target_shot_count=2, min_mapping_degree=0.9):
    """
    Run the pairing stage end-to-end and write fullmatch.csv. Each path argument
    defaults to `config` when None. Returns the output path.
    """
    shotsplits_dir = shotsplits_dir or config.SHOTSPLITS_DIR
    clusters_dir = clusters_dir or config.CLUSTERS_DIR
    out_path = out_path or config.FULLMATCH_OUT
    if data_csvs is None:
        data_csvs = [config.DATA_CSV] if config.DATA_CSV else list(config.BATCH_CSVS.values())

    data = load_data(data_csvs)
    shots = load_shots(shotsplits_dir)
    clusters = load_clusters(clusters_dir)

    clusters = clusters.merge(data[["brand_id", "ad_id", "ad_length"]])
    clusters = clusters.merge(shots[["ad_id", "shotsplit"]])

    clusters["ad_length"] = clusters["ad_length"].apply(standardize_ad_length)
    clusters = clusters[~clusters["ad_length"].isna()]

    # campaigns_dict: brand_id -> [{ad_id, length}] sorted by length
    campaigns_dict = defaultdict(list)
    for _, row in tqdm(clusters.iterrows(), total=len(clusters)):
        campaigns_dict[row["brand_id"]].append({"ad_id": row["ad_id"], "length": row["ad_length"]})
    for pair_id in campaigns_dict:
        campaigns_dict[pair_id].sort(key=lambda x: x["length"])

    cluster_dicts_by_pair = clusters.groupby("brand_id")["cluster"].first().to_dict()

    clusters["focal_to_targets"] = clusters.progress_apply(
        lambda row: process_row_reverse(row, campaigns_dict, cluster_dicts_by_pair),
        axis=1,
    )
    clusters["focal_to_targets"] = clusters["focal_to_targets"].progress_apply(sort_shot_keys)
    clusters["best_matching_ads"] = clusters["focal_to_targets"].progress_apply(find_best_matching_ads)
    pairs = unfold_best_matches(clusters)

    pairs["focal_shot_count"] = pairs["focal_shotsplit"].progress_apply(count_long_shots)
    pairs["target_shot_count"] = pairs["target_shotsplit"].progress_apply(count_long_shots)
    pairs["num_mapped_shots"] = pairs["mapping"].progress_apply(
        lambda d: len(set(v for v in d.values() if v is not None)))
    pairs["mapping_degree"] = pairs["num_mapped_shots"] / pairs["target_shot_count"]
    # Short(target)-video shots (>=5 frames) that did not map to any focal shot.
    pairs["unmatched_shots"] = pairs["target_shot_count"] - pairs["num_mapped_shots"]

    path = data[["ad_id", "relative_path"]]
    path.columns = ["focal_ad_id", "focal_relative_path"]
    pairs = pairs.merge(path)
    path = data[["ad_id", "relative_path"]]
    path.columns = ["target_ad_id", "target_relative_path"]
    pairs = pairs.merge(path)

    fullmatch = pairs[pairs["mapping_degree"] >= min_mapping_degree]
    fullmatch = fullmatch[fullmatch["focal_shot_count"] >= min_focal_shot_count]
    fullmatch = fullmatch[fullmatch["target_shot_count"] >= min_target_shot_count]
    fullmatch = fullmatch.drop_duplicates("focal_ad_id")

    # Record the gate that produced this file so the output is self-documenting.
    fullmatch = fullmatch.copy()
    fullmatch["min_mapping_degree"] = min_mapping_degree

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fullmatch.to_csv(out_path, index=False)
    print(f"[pairing] wrote {len(fullmatch)} rows -> {out_path}")
    return out_path


def build_parser():
    parser = argparse.ArgumentParser(description="Stage 3: pairing (shot overlap matching)")
    parser.add_argument("--shotsplits_dir", type=str, default=None)
    parser.add_argument("--clusters_dir", type=str, default=None)
    parser.add_argument("--data_csv", action="append", default=None,
                       help="Ad metadata CSV (repeatable). Defaults to config.")
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--min_focal_shot_count", type=int, default=4)
    parser.add_argument("--min_target_shot_count", type=int, default=2)
    parser.add_argument("--min_mapping_degree", type=float, default=config.MIN_MAPPING_DEGREE,
                       help="Keep pairs with mapping_degree >= this. Default comes from "
                            "config.yaml (params.min_mapping_degree, currently 0.9): recovers "
                            "true pairs split by fine-grained clustering, 0 false positives on "
                            "the 7-brand test set; 1.0 = exact fullmatch (over-tight).")
    return parser


def main():
    args = build_parser().parse_args()
    run_pairing(
        shotsplits_dir=args.shotsplits_dir,
        clusters_dir=args.clusters_dir,
        data_csvs=args.data_csv,
        out_path=args.out_path,
        min_focal_shot_count=args.min_focal_shot_count,
        min_target_shot_count=args.min_target_shot_count,
        min_mapping_degree=args.min_mapping_degree,
    )


if __name__ == "__main__":
    main()
