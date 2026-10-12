# video_curation

Collects brand ad videos from YouTube, pairs short and long cuts of the same ad, downloads them, and uploads them to the cluster.

## Setup

```bash
pip install -r requirements.txt
cp ../.env.example ../.env     # then fill in keys and DATA_ROOT
```

All paths come from the repo-root `.env` via `paths.py`:

| var | used for |
|---|---|
| `YOUTUBE_API_KEY` | YouTube Data API (`export_channel.py`, `channel_finder.py`) |
| `OPENAI_API_KEY` | embeddings (`initial_screening.py`) |
| `DATA_ROOT` | data folder, relative to the repo root or absolute (default `data`) |
| `DENO_PATH` | optional; deno.exe for yt-dlp if deno is not on `PATH` |
| `REMOTE_HOST`, `REMOTE_DIR` | upload target (`rsync_upload.py`, `transfer_file.py`) |

Layout under `DATA_ROOT`:

```
company_name/batchN.csv     input: brand_id, brand_name, featured_handle (NA/blank = skip)
video_metadata/             @handle.json, @handle.{pairs.csv, clustered.csv, dnwd.json}
videomedia/<Channel>/       <video_id>.mp4
logs/                       batchN.json, batchN_downloads.jsonl, batchN_videos.csv, batchN_failed.csv
```

`download.py` also needs Firefox logged in to YouTube (cookies are read from it).

## Run the pipeline

```bash
python pipeline.py --batch batch4
```

| step | code | does | writes |
|---|---|---|---|
| 1 | `export_channel.export_channel_uploads` | handle → channel ID → all uploads + metadata (YouTube Data API) | `video_metadata/@handle.json`, `logs/batch4.json` |
| 2 | `initial_screening.initial_screening` | keep videos < 70 s, OpenAI embeddings, DBSCAN clusters, pair short↔long (1.5–6× length, cos ≥ 0.75, ≤ 3 shorts per long) | `@handle.pairs.csv`, `@handle.clustered.csv`, `@handle.dnwd.json` |
| 3 | `update_log.post_screening_log_update` | adds screened video count | `logs/batch4.json` |
| 4 | `download.download_batch` | yt-dlp download of every video in `.dnwd.json`; 4 threads, 1 start / 1.8 s (`rate_limiter.py`), backs off 15 min × attempt on rate limit | `videomedia/<Channel>/<id>.mp4`, `logs/batch4_downloads.jsonl` |
| 5 | `update_log.post_download_log_update` | joins download log with metadata | `logs/batch4_videos.csv`, `logs/batch4_failed.csv` |

Reruns skip finished work: step 1 skips handles with an existing `.json`, step 2 skips handles with an existing `.dnwd.json`, step 4 skips IDs in `videomedia/_downloaded_ids.txt` or already on disk.

Steps can also run alone: `python initial_screening.py batch4`, `python download.py batch4`, `python update_log.py batch4`.

Pairing thresholds are constants in `initial_screening.run_clustering_from_json`; download settings are at the top of `download.py`.

## Standalone tools

| script | use |
|---|---|
| `channel_finder.py` | find a brand's channels: BFS over "featured channels" (depth 3) + channel search → `data/channels/<brand>.json` |
| `channel_review.py` | `streamlit run channel_review.py`; tick/untick channels from a `channel_finder` JSON, saves `flag.json`, exports a pruned list |
| `rsync_upload.py` | `python rsync_upload.py batch4` (WSL); rsyncs the folders in `logs/batch4_videos.csv` to `REMOTE_HOST:REMOTE_DIR`, then a `--checksum` verify pass |
| `transfer_file.py` | alternative upload (WSL): tar each `videomedia/` subfolder, send to `REMOTE_DIR/video_media`, extract remotely, verify count + size; resumable |
| `rate_limiter.py` | SQLite-backed rate limiter shared across processes; `python rate_limiter.py` runs a sanity check |

## Legacy

`code_archive/`: earlier download/pipeline/export versions, the pytube client (`youtube_setup.py`) and prototype notebooks (`initial_clustering.ipynb` and others). Nothing in the pipeline imports them.
