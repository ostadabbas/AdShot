import json
import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp
from yt_dlp.utils import DownloadError
from tqdm import tqdm

from rate_limiter import PersistentLeakyLimiter
from initial_screening import load_handles
from paths import METADATA_DIR, VIDEO_DIR, LOGS_DIR, batch_csv as resolve_batch_csv

# -------------------- CONFIG --------------------
DENO_PATH = os.getenv("DENO_PATH")  # optional; yt-dlp finds deno on PATH otherwise

MAX_WORKERS = 4         # tune: 4-8 is a common sweet spot before YouTube gets angry
BATCH_SIZE = 10         # how many tasks to launch per batch
MAX_RETRIES = 10

FORMAT = "137+140/best"
COOKIES_FROM_BROWSER = ("firefox",)
ARCHIVE_FILE = VIDEO_DIR / "_downloaded_ids.txt"
# ------------------------------------------------

archive_lock = threading.Lock()
archive_set = set()

def classify_ydl_error(e: Exception) -> str:
    m = str(e).lower()

    # ONLY retry this
    if ("rate-limited" in m) or \
       ("this content isn't available, try again later" in m):
        return "rate_limit"

    # Terminal / don't retry
    if "this video is private" in m:
        return "private"
    if "this live event will begin" in m:
        return "live_not_started"
    if "video unavailable" in m:
        return "unavailable"
    if "not made this video available in your country" in m:
        return "unavailable in account's country"
    if "live stream recording is not available" in m:
        return m

    return "other"

def sanitize_folder_name(name: str) -> str:
    if not name:
        return "unknown_channel"
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    name = name.strip().strip(".")
    return name[:120]

def extract_video_list(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("videos", "items", "data", "results"):
            v = payload.get(key)
            if isinstance(v, list):
                return v
        return [payload]
    return []

def get_video_id(rec: dict) -> str | None:
    for k in ("id", "video_id", "videoId", "videoID", "youtube_id"):
        v = rec.get(k)
        if v:
            return str(v).strip()
    return None

def get_channel_name(file_path: Path, payload, rec: dict) -> str:
    for src in (rec, payload if isinstance(payload, dict) else {}):
        for k in ("channel_name", "channelTitle", "channel", "handle", "brand_name"):
            v = src.get(k)
            if v:
                return str(v).strip()
    return file_path.name.removesuffix(".dnwd.json")

def find_video_path(out_dir: Path, video_id: str) -> str | None:
    """Find the downloaded video file: '<id>.*' or older '<duration>_<id>.*'."""
    for pattern in (f"{video_id}.*", f"*_{video_id}.*"):
        for p in out_dir.glob(pattern):
            return str(p)
    return None

def load_archive():
    global archive_set
    if ARCHIVE_FILE.exists():
        archive_set = set(x.strip() for x in ARCHIVE_FILE.read_text(encoding="utf-8").splitlines() if x.strip())

def mark_downloaded(video_id: str):
    # thread-safe append + set update
    with archive_lock:
        if video_id in archive_set:
            return
        archive_set.add(video_id)
        ARCHIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with ARCHIVE_FILE.open("a", encoding="utf-8") as f:
            f.write(video_id + "\n")

def download_one(video_id: str, out_dir: Path):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "format": FORMAT,
        "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
        "retries": 2,
        "fragment_retries": 2,
        "concurrent_fragment_downloads": 1,  # keep low since we're parallelizing at video-level
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if DENO_PATH:
        ydl_opts["js_runtimes"] = {"deno": {"path": DENO_PATH}}
    if COOKIES_FROM_BROWSER:
        ydl_opts["cookiesfrombrowser"] = COOKIES_FROM_BROWSER

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def load_logged_video_ids(log_file: Path) -> set:
    """Load video IDs already in the log file."""
    logged = set()
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    logged.add(json.loads(line.strip()).get("video_id"))
                except json.JSONDecodeError:
                    continue
    return logged

def worker_task(channel: str, video_id: str, limiter: PersistentLeakyLimiter) -> tuple[str, str | None]:
    """
    Returns (status, error_type)
    status in {"skip", "ok", "fail"}
    """
    out_dir = VIDEO_DIR / channel
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fast skip checks
    with archive_lock:
        if video_id in archive_set:
            return ("skip", None)

    if find_video_path(out_dir, video_id):
        mark_downloaded(video_id)
        return ("skip", "Downloaded Already")

    limiter.acquire()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            download_one(video_id, out_dir)
            mark_downloaded(video_id)
            return ("ok", None)

        except DownloadError as e:
            error_type = classify_ydl_error(e)
            # Retry ONLY rate-limit; YouTube may block "up to an hour"
            if error_type == "rate_limit" and attempt < MAX_RETRIES:
                time.sleep(900 * attempt)
                continue
            return ("fail", error_type)

    return ("fail", "rate_limit_not_resetting")

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def log_download(log_file: Path, logged_ids: set, handle: str, video_id: str, status: str, error_type: str | None, out_dir: Path):
    """Log a download result to the log file."""
    if video_id in logged_ids:
        return

    entry = {
        "handle": handle,
        "video_id": video_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S") if status == "ok" else None,
        "error_type": {"ok": None, "skip": "Already downloaded"}.get(status, error_type),
        "path": find_video_path(out_dir, video_id) if status != "fail" else None,
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logged_ids.add(video_id)

def download_batch(batch_csv=None):
    """
    Args:
        batch_csv (str or Path or None): Path to CSV of handles to process, or None to process all
    """
    handles = None
    batch_name = "all"
    if batch_csv:
        batch_name = Path(batch_csv).stem  # "batch1" from "batch1.csv"
        handles = load_handles(batch_csv)
        if not handles:
            print(f"No handles extracted from batch file: {batch_csv}")
            return

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    limiter = PersistentLeakyLimiter(VIDEO_DIR / "_rate_limit.sqlite", period_sec=1.8)
    load_archive()

    download_log_file = LOGS_DIR / f"{batch_name}_downloads.jsonl"
    logged_ids = load_logged_video_ids(download_log_file)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(METADATA_DIR.glob("*.dnwd.json"))
    if not files:
        print(f"No .dnwd.json files found in: {METADATA_DIR}")
        return

    tasks = []
    for fp in files:
        # dnwd file names expected as @handle.dnwd.json
        if handles is not None and fp.name.removesuffix(".dnwd.json") not in handles:
            continue

        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
            continue

        for rec in extract_video_list(payload):
            if not isinstance(rec, dict):
                continue
            vid = get_video_id(rec)
            if not vid:
                continue
            ch = sanitize_folder_name(get_channel_name(fp, payload, rec))
            tasks.append((ch, vid))

    # de-dupe tasks while preserving order
    tasks = list(dict.fromkeys(tasks))
    if not tasks:
        print("No download tasks found in metadata.")
        return

    counts = {"ok": 0, "skip": 0, "fail": 0}
    pbar = tqdm(total=len(tasks), desc="Processed", unit="video")

    for batch in chunked(tasks, BATCH_SIZE):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(worker_task, ch, vid, limiter): (ch, vid) for ch, vid in batch}

            for fut in as_completed(futures):
                ch, vid = futures[fut]
                try:
                    status, error_type = fut.result()
                except Exception as e:
                    status, error_type = "fail", f"exception: {e}"

                counts[status] += 1
                log_download(download_log_file, logged_ids, ch, vid, status, error_type, VIDEO_DIR / ch)
                pbar.update(1)
                pbar.set_postfix(**counts)

    pbar.close()

    print(f"Done. ok={counts['ok']}, skipped={counts['skip']}, failed={counts['fail']}")
    return download_log_file

if __name__ == "__main__":
    # python download.py batch2
    download_batch(resolve_batch_csv(sys.argv[1]))
