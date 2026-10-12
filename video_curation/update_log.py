import json
import glob
import os
import csv
from pathlib import Path

from paths import METADATA_DIR, DATA_ROOT, LOGS_DIR


def post_screening_log_update(channel_logs_path, video_metadata_glob):
    """
    Update 'Initial Screening' in channel_logs based on video metadata files.
    
    Args:
        channel_logs_path: Path to the channel_logs.json file
        video_metadata_glob: Glob pattern for video metadata files (e.g., "*.dnwd.json")
    
    Returns:
        dict with 'updated', 'total_entries', and 'metadata_files' counts
    """
    with open(channel_logs_path, "r", encoding="utf-8") as f:
        channel_logs = json.load(f)

    handle_to_video_count = {}
    for vm_path in glob.glob(video_metadata_glob):
        base = os.path.basename(vm_path)
        if base.endswith(".dnwd.json"):
            handle = base[:-len(".dnwd.json")]
        else:
            handle = base.split('.')[0]
        
        try:
            with open(vm_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                count = len(data.get("videos", []))
        except json.JSONDecodeError:
            count = 0
        
        handle_to_video_count[handle] = count

    updated_count = 0
    for entry in channel_logs:
        handle = entry.get("handle")
        if handle and handle in handle_to_video_count:
            if "stage" not in entry or not isinstance(entry["stage"], dict):
                entry["stage"] = {}
            entry["stage"]["Initial Screening"] = handle_to_video_count[handle]
            updated_count += 1

    with open(channel_logs_path, "w", encoding="utf-8") as f:
        json.dump(channel_logs, f, indent=2, ensure_ascii=False)

    return {
        "updated": updated_count,
        "total_entries": len(channel_logs),
        "metadata_files": len(handle_to_video_count)
    }

def post_download_log_update(
    batch_csv,
    video_metadata_dir=METADATA_DIR,
    data_root=DATA_ROOT,
    logs_dir=LOGS_DIR,
):
    """
    Main function to process batch CSV and produce ok and failed video logs.
    Args:
        batch_csv: str, path to the batch company_name csv.
        video_metadata_dir: str, directory with .dnwd.json files.
        data_root: str, root data path for relative calculations.
        logs_dir: str, directory to write ok/failed logs.
    Outputs:
        Writes <BATCH>_videos.csv and <BATCH>_failed.csv to logs_dir.
    """
    # Derive BATCH from filename, e.g. batch4 from .../batch4.csv
    batch_prefix = Path(batch_csv).stem

    dnwd_log = os.path.join(logs_dir, f"{batch_prefix}_downloads.jsonl")
    ok_csv = os.path.join(logs_dir, f"{batch_prefix}_videos.csv")
    failed_csv = os.path.join(logs_dir, f"{batch_prefix}_failed.csv")

    # Helpers
    def build_handle_map(batch_csv_path: str):
        """
        Returns:
          handle_to_brand: {featured_handle: (brand_id, brand_name)}
          handles: list of featured_handle in file order
        """
        handle_to_brand = {}
        handles = []

        with open(batch_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = (row.get("featured_handle") or "").strip()
                if not h or h == "NA":
                    continue

                bid = (row.get("brand_id") or "").strip()
                bname = (row.get("brand_name") or "").strip()

                if h not in handle_to_brand:
                    handle_to_brand[h] = (bid, bname)
                    handles.append(h)

        return handle_to_brand, handles

    def parse_downloads(jsonl_path: str, data_root_abs: str):
        """
        Reads download JSONL and returns:
          ok_index:     {video_id: rel_path_from_data}
          failed_rows:  list of dicts for failures
        A line is treated as OK if it has a usable 'path'.
        Otherwise it's a failure (or if error_type exists and you prefer).
        """
        ok_index = {}
        failed_rows = []

        if not os.path.exists(jsonl_path):
            print(f"[WARN] download log not found: {jsonl_path}")
            return ok_index, failed_rows

        data_root_abs = os.path.normpath(data_root_abs)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                vid = rec.get("video_id") or ""
                handle = rec.get("handle") or ""
                error_type = rec.get("error_type") or ""
                ts = rec.get("timestamp")

                abs_path = rec.get("path")

                # Treat as OK only if path exists and is non-empty
                if abs_path:
                    abs_path = os.path.normpath(abs_path)
                    try:
                        rel = os.path.relpath(abs_path, data_root_abs).replace("\\", "/")
                    except ValueError:
                        # different drive etc -> consider failure
                        failed_rows.append({
                            "handle": handle,
                            "video_id": vid,
                            "error_type": error_type or "Path on different drive",
                            "timestamp": ts,
                            "path": abs_path,
                        })
                        continue

                    ok_index[vid] = rel
                else:
                    # failure / skipped / blocked / etc.
                    failed_rows.append({
                        "handle": handle,
                        "video_id": vid,
                        "error_type": error_type or "No path",
                        "timestamp": ts,
                        "path": "",
                    })

        return ok_index, failed_rows

    # Main logic
    handle_to_brand, handles = build_handle_map(batch_csv)
    ok_index, failed_rows = parse_downloads(dnwd_log, data_root)

    ok_header = [
        "brand_id", "brand_name", "ad_id", "ad_length", "ad_title",
        "publishedAt", "viewCount", "likeCount", "channelTitle", "relative_path"
    ]

    failed_header = [
        "handle", "brand_id", "brand_name", "video_id", "error_type", "timestamp", "path"
    ]

    ok_rows = []
    missing_meta = 0
    skipped_not_downloaded = 0

    for handle in handles:
        meta_file = os.path.join(video_metadata_dir, f"{handle}.dnwd.json")
        if not os.path.exists(meta_file):
            missing_meta += 1
            continue

        with open(meta_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                continue

        videos = data.get("videos", []) if isinstance(data, dict) else data
        if not isinstance(videos, list):
            continue

        brand_id, brand_name = handle_to_brand.get(handle, ("", ""))

        for v in videos:
            if not isinstance(v, dict):
                continue

            ad_id = v.get("id", "")
            if not ad_id:
                continue

            # ✅ Only keep rows that are actually in download log with a path
            rel_path = ok_index.get(ad_id)
            if not rel_path:
                skipped_not_downloaded += 1
                continue

            ok_rows.append([
                brand_id, brand_name,
                ad_id,
                v.get("video_duration", ""),
                v.get("title", ""),
                v.get("publishedAt", ""),
                v.get("viewCount", ""),
                v.get("likeCount", ""),
                v.get("channelTitle", ""),
                rel_path
            ])

    # enrich failed rows with brand info
    for fr in failed_rows:
        bid, bname = handle_to_brand.get(fr["handle"], ("", ""))
        fr["brand_id"] = bid
        fr["brand_name"] = bname

    # Write outputs
    os.makedirs(os.path.dirname(ok_csv), exist_ok=True)

    with open(ok_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(ok_header)
        w.writerows(ok_rows)

    with open(failed_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=failed_header)
        w.writeheader()
        w.writerows(failed_rows)

    print(f"OK rows written: {len(ok_rows)} -> {ok_csv}")
    print(f"Failures written: {len(failed_rows)} -> {failed_csv}")
    print(f"Missing meta files: {missing_meta}")
    print(f"Skipped (present in meta but not downloaded OK): {skipped_not_downloaded}")
    print(f"Download log OK entries: {len(ok_index)}")
    # Optionally return stats, paths, etc.:
    return {
        "ok_csv": ok_csv,
        "failed_csv": failed_csv,
        "n_ok": len(ok_rows),
        "n_failed": len(failed_rows),
        "missing_meta": missing_meta,
        "skipped_not_downloaded": skipped_not_downloaded,
        "download_log_ok_entries": len(ok_index),
    }
    
    
if __name__ == "__main__":
    # python update_log.py batch2
    import sys
    result = post_screening_log_update(
        LOGS_DIR / f"{sys.argv[1]}.json",
        os.path.join(METADATA_DIR, "*.dnwd.json")
    )
    print(f"Updated {result['updated']} of {result['total_entries']} entries")
