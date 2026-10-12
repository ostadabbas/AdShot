# Clustering & Pairing — Thresholds and Switches

Inventory of every tunable that affects which pairs come out of the pipeline,
scoped to the **clustering** (`s2_clustering.py`) and **pairing** (`s3_pairing.py`)
stages. Line numbers are current as of this writing; treat them as anchors, not
guarantees.

## Why this matters

The clustering stage decides how finely shots are grouped; the pairing stage
turns shared-cluster membership into shot-to-shot mappings and filters them. Two
knobs dominate the output:

- **C1 (clustering eps selection)** — the real driver of recall. Finer clustering
  splits a video's shots across more clusters, so a correct pair's shots stop
  sharing clusters and its `mapping_degree` falls below the keep-gate.
- **P1 (`mapping_degree` cutoff)** — the blunt compensator. Lowering it recovers
  pairs that clustering split, at the risk of admitting junk.

Observed on the 7-brand test set (embeddings byte-identical to the reference,
cosine 1.0): with C1 = "first mode" and P1 = 1.0 → 4 pairs; P1 = 0.9 → 11 pairs
(all visually correct, including 2 not present in the legacy `fullmatch823.csv`).
`fullmatch823.csv` is therefore a recall reference, not an exact-match oracle.

---

## Clustering — `s2_clustering.py`

| # | Knob | Loc | Current | Effect |
|---|---|---|---|---|
| C1 | **eps selection: "select FIRST mode"** | L271–273 | first (lowest-eps) silhouette peak | Root-cause switch. Picks the lowest-eps local max → finer clusters → splits shots → drops `mapping_degree`. Alternatives: max silhouette, or a higher mode = coarser clustering. |
| C2 | `eps_range` | L204 | `np.linspace(0.05, 0.9, 35)` | The eps grid scanned (step 0.025). Sets bounds + resolution of what eps is selectable. |
| C3 | `min_samples` | L190 | `1` | DBSCAN min core size. =1 → no noise points; every shot joins a cluster. |
| C4 | `metric` | L221 | `'cosine'` | Distance metric (matches the L2-normalized embeddings). |
| C5 | silhouette validity window | L225 | `2 <= n_clusters < total_shots` | Only scores eps values yielding a sane cluster count; others get silhouette = −1. |
| C6 | `find_peaks(prominence, distance)` | L261–265 | `prominence=0.01, distance=2` | Defines what counts as a "mode" in the silhouette curve. Higher prominence = fewer, stronger modes. |
| C7 | valid-peak floor | L268 | `silhouette > 0` | Discards non-positive-silhouette peaks. |
| C8 | `giant_ratio` fallback gate | L229, L280 | `< 0.5` | In the no-peak fallback, rejects clusterings where one cluster holds >50% of shots. |
| C9 | fallback chain | L280–287 | max-sil w/ giant<0.5 → else max-sil | Only used when no modes are found. |
| C10 | embedding normalization | L110–116 | L2 normalize | Fixed; defines the cosine geometry. |
| C11 | shot set fed to clustering | L47–80 | from `*_pNN.jpg` keyframe filenames | Which shots get clustered = which have keyframes. Indirectly sets `total_shots`, and thus the silhouette curve and selected eps. |

## Pairing — `s3_pairing.py`

| # | Knob | Loc | Current | Effect |
|---|---|---|---|---|
| P1 | **`min_mapping_degree`** | L256 | `0.9` (CLI `--min_mapping_degree`) — **base-pipeline default** | The keep-gate. `mapping_degree = num_mapped_shots / target_shot_count`. 0.9 recovers true pairs that fine clustering splits by one shot; 0 false positives on the 7-brand test set. 1.0 = exact fullmatch (over-tight). |
| P2 | `min_focal_shot_count` | L257 | `4` (CLI) | Drops focal ads with fewer than 4 "long" shots. |
| P3 | `min_target_shot_count` | L258 | `2` (CLI) | Drops targets with fewer than 2 "long" shots. |
| P4 | `count_long_shots` frame floor | L198 | `l >= 5` frames | Defines a "real" shot → sets both shot counts and the `mapping_degree` denominator. |
| P5 | `standardize_ad_length` buckets | L82–90 | `[14,15,16]→15`, `[29,30,31]→30`, else drop | Only ~15s / ~30s ads survive to pairing. |
| P6 | candidate direction | L111 | `ad["length"] < focal length` (strict) | Targets must be strictly shorter than the focal. |
| P7 | best-match rule | L152–162 | max unique-target overlap; ties kept; `max_overlap > 0` | How the focal→target winner(s) are chosen. |
| P8 | `drop_duplicates("focal_ad_id")` | L259 | first kept | One row per focal ad; order-dependent. |
| P9 | shot de-dup on load | L53 | `drop_duplicates("ad_id", keep="first")` | Dedups shotsplits if an ad appears twice. |

---

## Helper scripts (this folder)

- `pair_viz.py` — per chosen pair, render a 2-row montage: top = each SHORT
  (target) shot's mid frame (p50 keyframe), bottom = the corresponding LONG
  (focal) shot it maps to. Use it to judge precision/recall by eye.
  `--fullmatch_csv --keyframes_dir --output_dir [--percentile 50] [--brand N]`


____
min_mapping_degree=0.9