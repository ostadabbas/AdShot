"""
Run one batch end to end:

    python pipeline.py --batch batch4

1. export_channel_uploads     -> video_metadata/@handle.json, logs/{batch}.json
2. initial_screening          -> video_metadata/@handle.{pairs.csv, clustered.csv, dnwd.json}
3. post_screening_log_update  -> logs/{batch}.json
4. download_batch             -> videomedia/<Channel>/<id>.mp4, logs/{batch}_downloads.jsonl
5. post_download_log_update   -> logs/{batch}_videos.csv, logs/{batch}_failed.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from paths import METADATA_DIR, LOGS_DIR, COMPANY_DIR, batch_csv  # also loads .env
from export_channel import export_channel_uploads
from initial_screening import initial_screening
from update_log import post_screening_log_update, post_download_log_update
from download import download_batch


def generate_channel_logs(batch_csv_file, metadata_dir, logs_dir):
    """Export uploads for every handle in the batch and record per-brand video counts."""
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    channel_logs_path = Path(logs_dir) / f"{Path(batch_csv_file).stem}.json"
    channel_logs_path.parent.mkdir(parents=True, exist_ok=True)

    channel_logs = []
    with open(batch_csv_file, newline='', encoding='utf-8') as csvfile:
        for row in csv.DictReader(csvfile):
            handle = (row.get("featured_handle") or "").strip()
            ch_log = {
                "brand_name": row.get("brand_name"),
                "handle": handle,
                "stage": {
                    "Videos": 0,
                    "Initial Screening": "No",
                    "Video Downloaded": 0
                }
            }
            channel_logs.append(ch_log)
            if not handle or handle.upper() == "NA":
                continue

            videos_path = metadata_dir / f"{handle}.json"
            if not videos_path.is_file():
                try:
                    export_channel_uploads(handle, str(videos_path))
                except Exception as e:
                    print(f"[WARN] export failed for {handle}: {e}")

            if videos_path.is_file():
                try:
                    obj = json.loads(videos_path.read_text(encoding="utf-8"))
                    ch_log["stage"]["Videos"] = len(obj.get("videos") or [])
                except Exception:
                    pass

    with open(channel_logs_path, "w", encoding="utf-8") as f:
        json.dump(channel_logs, f, indent=2, ensure_ascii=False)
    return channel_logs_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline: Process a given company's batch CSV")
    parser.add_argument("--batch", required=True, type=str, help="Batch name (e.g. batch4) or batch csv file (e.g. batch4.csv)")
    args = parser.parse_args()

    batch_csv_file = batch_csv(args.batch)
    if not batch_csv_file.is_file():
        print(f"[ERROR] Batch CSV file does not exist: {batch_csv_file}")
        if not COMPANY_DIR.is_dir():
            print(f"[ERROR] Company directory does not exist: {COMPANY_DIR}")
        else:
            csvs = [f.name for f in COMPANY_DIR.glob("*.csv")]
            if csvs:
                print("Available batch CSV files in this folder:")
                for f in csvs:
                    print(f"  - {f}")
            else:
                print("(No CSV files found in this folder)")
        sys.exit(1)

    channel_logs_path = generate_channel_logs(batch_csv_file, METADATA_DIR, LOGS_DIR)
    initial_screening(handles_csv=batch_csv_file, metadata_dir=METADATA_DIR)
    post_screening_log_update(channel_logs_path, os.path.join(METADATA_DIR, "*.dnwd.json"))
    download_batch(batch_csv_file)
    post_download_log_update(batch_csv_file)
