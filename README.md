# AdShot: Benchmarking Multimodal LLMs for Video Advertisement Clipping

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?logo=arxiv&logoColor=white)](#)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-adshot-ffd21e)](https://huggingface.co/datasets/omrastogi/adshot_benchmark)
[![GitHub](https://img.shields.io/badge/GitHub-AdShot-181717?logo=github)](https://github.com/ostadabbas/AdShot)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)

**AdShot** is the first large-scale benchmark for **ad shot selection**: given the shots of a
30-second TV ad, choose the subset that composes its professionally edited 15-second cut.
It contains **823 ad pairs** (1,646 videos) from **194 brands** across **17 industries**, each
annotated with shot-level boundaries, a focal→target shot mapping, and 18 content-feature labels.

![Shot selection: shots of a 30 s ad mapped to its 15 s cut](asset/shot_selection.gif)

- 📄 **Paper:** coming soon
- 🤗 **Dataset:** [huggingface.co/datasets/omrastogi/adshot_benchmark](https://huggingface.co/datasets/omrastogi/adshot_benchmark) *(gated)*
- 💻 **Code:** [github.com/ostadabbas/AdShot](https://github.com/ostadabbas/AdShot)

---

## Why AdShot

Advertisers cut multiple duration variants of each campaign; producing the shorter version is
skilled, labor-intensive editorial work. AdShot reframes that work as a learnable
**set-prediction task** and measures how well MLLMs reproduce real professional edits under a
strict duration budget, rather than testing comprehension through question answering.

---

## Access and License

We **do not redistribute raw video**. Rights to the original ads remain with their brand owners;
AdShot ships annotations plus derived per-shot media for the focal (30 s) ads only.

| file | where | access | contents |
|---|---|---|---|
| `AdShot823.csv` | this repo / HF (`adshot823.csv`) | open | 823 pairs: YouTube IDs, shot boundaries, focal→target mapping |
| `keyframes_k5_adshot823.h5` | HF, 16.75 GB | gated | 5 RGB keyframes per shot, 819 focal ads |
| `audio_16k_adshot823.h5` | HF, 1.60 GB | gated | 16 kHz mono audio per shot, 823 focal ads |
| raw `.mp4`, target-ad media | — | not released | — |

**Gated files** require accepting the Data Use Agreement on the
[dataset page](https://huggingface.co/datasets/omrastogi/adshot_benchmark). By requesting access you agree to:

- use AdShot only for **non-commercial academic research**;
- **not redistribute** or re-host the data, including to lab members without their own access;
- not reconstruct or publish the original ads or their audio, or use the data to identify individuals;
- show only individual frames in publications, and only when necessary;
- **delete** all copies on project completion or on request from the authors;
- **cite** the AdShot paper. Access is revocable.

**CC BY-NC 4.0** covers the annotations and packaging. No license is granted to the underlying ads.

### Raw video

You don't need raw video to run the benchmark: the released keyframes and audio are the model
inputs. `focal_relative_path` / `target_relative_path` are paths from our build
(`videomedia/<Channel>/<id>.mp4`) and won't exist on your machine. If you need source video,
fetch it yourself by YouTube ID (e.g. with `yt-dlp`) under YouTube's Terms of Service. IDs
disappear over time; four focal videos are already gone and have no keyframes:
`dujob0gpAg4`, `qQLPVjLNGUA`, `3yBzrFMnQa0`, `hET1H6RZVwQ`.

---

## Dataset

### `AdShot823.csv`

823 focal (30 s) → target (15 s) pairs where every target shot maps to a focal shot.

| column | meaning |
|---|---|
| `brand_id` | brand |
| `focal_ad_id`, `target_ad_id` | YouTube video IDs (30 s, 15 s) |
| `focal_ad_length`, `target_ad_length` | seconds |
| `focal_shotsplit`, `target_shotsplit` | inclusive end-frame index of each shot: shot 1 is `[0, split[0]]`, shot 2 is `[split[0]+1, split[1]]` |
| `focal_shot_count`, `target_shot_count`, `num_mapped_shots` | shot counts |
| `mapping` | focal shot → target shot, `None` if dropped in the 15 s cut: `{'shot_01': 'shot_01', 'shot_02': None, ...}` |
| `mapping_degree` | `num_mapped_shots / target_shot_count`; `1.0` for every row (full matches only) |
| `focal_relative_path`, `target_relative_path` | build paths, see [Raw video](#raw-video) |

### HDF5 media

```
keyframes_k5_adshot823.h5
  /<focal_ad_id>            uint8 (n_shots, 5, H, W, 3), lzf
      attrs: shot_durations, relative_path
  frames at the 10/25/50/75/90th percentile of each shot, pre-resized with Qwen smart_resize(max_pixels=151200)

audio_16k_adshot823.h5
  /<focal_ad_id>/shot_01 … shot_NN   float32 (n_samples,), mono, sr=16000
```

The keyframe array can hold a couple of extra boundary shots, so slice to `focal_shot_count`.

### Loading

```python
from huggingface_hub import hf_hub_download
import h5py, pandas as pd

repo = "omrastogi/adshot_benchmark"
kf  = hf_hub_download(repo, "keyframes_k5_adshot823.h5", repo_type="dataset")
csv = hf_hub_download(repo, "adshot823.csv", repo_type="dataset")

row = pd.read_csv(csv).iloc[0]
with h5py.File(kf, "r") as f:
    if row.focal_ad_id in f:                                       # 4 ads have no keyframes
        shots = f[row.focal_ad_id][:int(row.focal_shot_count)]     # (n, 5, H, W, 3)
```

---

## Task and Evaluation

**Input:** task instruction, per-shot durations, and the shots' keyframes (plus per-shot audio
for audio-visual models). Prompts are built in `evaluations/build_message.py` and
`evaluations/build_message_audio.py`.

**Output:** a JSON object with exactly these fields:

```json
{
  "selected_shots": ["shot4", "shot7", "shot8", "shot15"],
  "total_length_sec": 14.76,
  "explanations": ["reason for shot4", "reason for shot7", "reason for shot8", "reason for shot15"]
}
```

Selected shots must be **whole shots** totalling **14–16 s** (target 15 s).

**Metrics.** Selection is scored as set prediction of the predicted shots `S` against the
ground-truth target shots `S*` (focal shots with a non-`None` mapping):

- **Precision** = `|S ∩ S*| / |S|` (penalizes over-selection)
- **Recall** = `|S ∩ S*| / |S*|` (penalizes missed shots)
- **F1**, harmonic mean of the two
- **IoU** = `|S ∩ S*| / |S ∪ S*|`
- **MDE** = `E[|T − 15|]`, mean duration error of the selected total `T`, with signed `MDEpos` / `MDEneg` for over- and under-shoot

The evaluation scripts also report per-shot accuracy and AUC.

### Evaluated models

13 video-only MLLMs: LLaVA-OneVision (7B, 72B), Qwen2.5-VL (7B, 32B), Qwen3-VL (8B, 32B),
InternVL2.5 (8B, 26B), InternVL3.5 (8B, 38B), Aria (25.3B), MiniCPM-V (8B), Molmo2 (8B).
2 audio-visual MLLMs: Qwen2.5-Omni (7B), MiniCPM-o 2.6 (8B).

---

## Repository

```
video_curation/   ──►  pair_extraction/  ──►  AdShot823.csv  ──►  evaluations/    (zero-shot MLLM benchmark)
 videos +               shot detection,        823 pairs     └──►  qwen_pipeline/  (Qwen3-VL batch eval + LoRA SFT)
 batchN_videos.csv      embeddings, pairing
```

| folder | what it does | runs on | docs |
|---|---|---|---|
| `video_curation/` | Finds brand channels, lists uploads, screens for short/long ad candidates (OpenAI embeddings + DBSCAN), downloads with yt-dlp, uploads to the cluster. Outputs `videomedia/<Channel>/<id>.mp4` and `logs/batchN_videos.csv`. | local (Windows/WSL) | [readme](video_curation/readme.md) |
| `pair_extraction/` | Per brand: AutoShot shot detection → keyframes → Swin3D-B embeddings → DBSCAN → shot-overlap matching. Outputs `fullmatch.csv` (the `AdShot823.csv` schema). | cluster, `shot` env | [readme](pair_extraction/readme.md), `THRESHOLDS.md` |
| `evaluations/` | Zero-shot evaluation of video and audio-visual MLLMs on shot selection. | cluster, `mllm` / `mllm2` envs | [readme](evaluations/readme.md) |
| `qwen_pipeline/` | Train/test split, keyframe cache (HDF5), batched Qwen3-VL-8B inference, LoRA fine-tuning. Reuses the `evaluations/` prompts and metrics. | cluster, `mllm2` env | [training readme](qwen_pipeline/training/README.md), `results/SFT_EVAL_SUMMARY.md` |

### Setup

```bash
cp .env.example .env    # API keys, DATA_ROOT, upload target, HF_TOKEN
```

`.env` is gitignored. `video_curation` reads it through `video_curation/paths.py`. Cluster
scripts expect `HF_TOKEN` exported in your shell before `sbatch` (sbatch passes your
environment through). Environment and install steps for each part are in its folder's readme.

---

## Citation

```bibtex
@inproceedings{xie2026adshot,
  title     = {AdShot: Benchmarking Multimodal Large Language Models for Video Advertisement Clipping},
  author    = {Xie, Wen and Rastogi, Om and Rangoju, Sai Siddhartha Vivek Dhir and
               Bart, Yakov and Ostadabbas, Sarah and Overgoor, Gijs},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2026}
}
```

## Contact

Open an issue on [GitHub](https://github.com/ostadabbas/AdShot). Augmented Cognition Lab, Northeastern University.
