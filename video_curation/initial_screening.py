#!/usr/bin/env python3
"""
initial_screening.py

Screens each channel's uploads down to likely short/long ad pairs.

Per @handle.json (written by export_channel.py):
1) Keep videos shorter than DURATION_THRESHOLD seconds
2) Embed "title + channel + description" with OpenAI (text-embedding-3-small)
3) Cluster embeddings with DBSCAN (cosine distance)
4) Within each cluster, match short<->long pairs using:
   - duration ratio bounds
   - cosine similarity threshold
   - capacity constraint per long video
5) Write @handle.pairs.csv, @handle.clustered.csv and @handle.dnwd.json (paired videos only)

Environment:
- OPENAI_API_KEY in your shell or in a local .env file.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

PAIR_COLUMNS = [
    "cluster",
    "short_id", "long_id",
    "cos_sim",
    "short_dur", "long_dur",
    "short_title", "long_title",
    "ratio",
]


# -----------------------------
# Text formatting / embeddings
# -----------------------------
def row_to_text(row: pd.Series) -> str:
    """Convert a dataframe row to a text prompt for embedding."""
    title = str(row.get("title", "") or "")
    desc = str(row.get("description", "") or "")
    ch_title = str(row.get("channelTitle", "") or "")
    handle = str(row.get("handle", "") or "")
    if handle and not handle.startswith("@"):
        handle = "@" + handle

    return (
        f"Title: {title}\n"
        f"Channel: {ch_title} ({handle})\n"
        f"Description: {desc}"
    )


def _batched(seq: Sequence[str], batch_size: int) -> Sequence[Sequence[str]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def embed_texts_openai(
    texts: List[str],
    model: str,
    batch_size: int,
    max_retries: int = 5,
    sleep_base_s: float = 1.5,
) -> List[List[float]]:
    """
    Create embeddings for a list of texts using the OpenAI embeddings endpoint.
    Uses batching + basic exponential backoff retries.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found (set it in env or a .env file).")

    client = OpenAI(api_key=api_key)

    out: List[List[float]] = []
    for batch in _batched(texts, batch_size=batch_size):
        for attempt in range(max_retries + 1):
            try:
                resp = client.embeddings.create(model=model, input=list(batch))
                out.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if attempt >= max_retries:
                    raise
                sleep_s = sleep_base_s * (2 ** attempt)
                print(f"[warn] embedding batch failed (attempt {attempt+1}/{max_retries}). "
                      f"Sleeping {sleep_s:.1f}s. Error: {e}")
                time.sleep(sleep_s)

    return out


# -----------------------------
# Pair matching logic
# -----------------------------
def match_pairs_within_cluster_relative(
    dfc: pd.DataFrame,
    ratio_min: float = 1.5,
    ratio_max: float = 6.0,
    min_cos: float = 0.75,
    long_capacity: int = 3,
    top_k_per_short: int = 5,
) -> pd.DataFrame:
    """
    Within a cluster, match each short video to (at most) one longer video, while allowing
    each long video to be used up to `long_capacity` times.

    Rules:
    - long_duration > short_duration
    - ratio in [ratio_min, ratio_max]
    - cosine similarity >= min_cos
    - greedy selection by cosine similarity, with long capacity constraint
    """
    dfc = dfc.reset_index(drop=True)
    n = len(dfc)
    if n < 2:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    X = np.vstack(dfc["embedding"].values)
    cos = cosine_similarity(X, X)

    dur = dfc["video_duration"].to_numpy(dtype=float)
    di = dur[:, None]
    dj = dur[None, :]

    ratio = dj / np.maximum(di, 1.0)
    valid = (dj > di) & (ratio >= ratio_min) & (ratio <= ratio_max)

    # Only score valid pairs; invalid gets a very negative score
    score = np.where(valid, cos, -1e9)

    edges: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in np.argsort(-score[i])[:top_k_per_short]:
            s = float(score[i, j])
            if s >= min_cos:
                edges.append((i, j, s))

    edges.sort(key=lambda x: x[2], reverse=True)
    cap = {j: int(long_capacity) for j in range(n)}
    used_short = set()
    pairs = []

    for i, j, s in edges:
        if i in used_short or cap[j] <= 0:
            continue
        used_short.add(i)
        cap[j] -= 1

        short_d, long_d = dur[i], dur[j]
        pairs.append({
            "cluster": dfc["cluster"].iloc[0],
            "short_id": str(dfc.loc[i, "id"]),
            "long_id":  str(dfc.loc[j, "id"]),
            "cos_sim":  s,
            "short_dur": int(short_d),
            "long_dur":  int(long_d),
            "short_title": str(dfc.loc[i, "title"]),
            "long_title":  str(dfc.loc[j, "title"]),
            "ratio": float(long_d / max(short_d, 1.0)),
        })

    if not pairs:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    return pd.DataFrame(pairs).sort_values("cos_sim", ascending=False).reset_index(drop=True)


def match_all_clusters_relative(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    outs = [match_pairs_within_cluster_relative(dfc, **kwargs) for _, dfc in df.groupby("cluster")]
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame(columns=PAIR_COLUMNS)


def run_clustering_from_json(json_file: str) -> pd.DataFrame:
    # ---- hardcoded config (edit as you like) ----
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBED_BATCH = 64

    EPS = 0.2
    MIN_SAMPLES = 2

    RATIO_MIN = 1.5
    RATIO_MAX = 6.0
    MIN_COS = 0.75
    LONG_CAPACITY = 3
    TOP_K_PER_SHORT = 5
    DURATION_THRESHOLD = 70

    PAIRS_OUT = Path(json_file).with_suffix(".pairs.csv")
    CLUSTERED_OUT = Path(json_file).with_suffix(".clustered.csv")
    DNWD_JSON_OUT = Path(json_file).with_name(Path(json_file).stem + ".dnwd.json")
    # --------------------------------------------

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data.get("videos", []))
    df["handle"] = data.get("seed", "") or ""
    for col in ("title", "description", "channelTitle"):
        if col not in df.columns:
            df[col] = ""
    df["video_duration"] = pd.to_numeric(df.get("video_duration"), errors="coerce")

    # Keep Duration below threshold
    df = df[df["video_duration"] < DURATION_THRESHOLD].copy()

    if len(df) == 0:
        print("No videos after filtering. Skipping clustering and pairing.")
        pairs = pd.DataFrame(columns=PAIR_COLUMNS)
    else:
        texts = df.apply(row_to_text, axis=1).tolist()
        df["embedding"] = embed_texts_openai(texts=texts, model=EMBEDDING_MODEL, batch_size=EMBED_BATCH)

        X = np.vstack(df["embedding"].values).astype(np.float32)
        df["cluster"] = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="cosine", n_jobs=-1).fit_predict(X)

        print("\nTop clusters by size:")
        print(df["cluster"].value_counts().head(10))

        pairs = match_all_clusters_relative(
            df,
            ratio_min=RATIO_MIN,
            ratio_max=RATIO_MAX,
            min_cos=MIN_COS,
            long_capacity=LONG_CAPACITY,
            top_k_per_short=TOP_K_PER_SHORT,
        )

    pairs.to_csv(PAIRS_OUT, index=False)
    print(f"\nWrote pairs: {PAIRS_OUT}  (rows={len(pairs)})")

    # dnwd json: same structure as input, only paired videos
    paired_ids = set(pairs["short_id"].dropna().astype(str)) | set(pairs["long_id"].dropna().astype(str))
    out = dict(data)
    out["videos"] = [v for v in data.get("videos", []) if str(v.get("id")) in paired_ids]
    with open(DNWD_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote dnwd JSON: {DNWD_JSON_OUT}  (videos={len(out['videos'])})")

    df.drop(columns=["embedding"], errors="ignore").to_csv(CLUSTERED_OUT, index=False)
    print(f"Wrote clustered dataframe: {CLUSTERED_OUT}")

    return pairs


def load_handles(handles_csv: str) -> set[str]:
    """Return the set of '@handle' values in a batch CSV's featured_handle column."""
    handles = set()
    with open(handles_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            h = (row.get("featured_handle") or "").strip()
            if h.startswith("@"):
                handles.add(h)
    return handles


def initial_screening(handles_csv: str, metadata_dir: str):
    metadata_dir = Path(metadata_dir)

    for handle in sorted(load_handles(handles_csv)):
        json_path = metadata_dir / f"{handle}.json"
        if not json_path.exists():
            print(f"Missing JSON for {handle}: {json_path}")
            continue

        if json_path.with_name(json_path.stem + ".dnwd.json").exists():
            print(f"Found dnwd JSON for {json_path}, skipping.")
            continue

        print(f"\nProcessing {json_path}")
        try:
            run_clustering_from_json(str(json_path))
        except Exception as e:
            print(f"ERROR processing {json_path}: {e}")


if __name__ == "__main__":
    # python initial_screening.py batch2
    import sys
    from paths import METADATA_DIR, batch_csv

    initial_screening(handles_csv=batch_csv(sys.argv[1]), metadata_dir=METADATA_DIR)
