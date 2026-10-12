import json
import random
import time
from pathlib import Path

import yt_dlp
from tqdm import tqdm

def skewed_sleep():
    if random.random() < 0.90:                 # 90% of the time: short
        return random.uniform(0.0, 0.5)
    else:                                      # 10% of the time: long
        return random.uniform(3.0, 6.0)

# -------------------- CONFIG --------------------
METADATA_DIR = Path(r"E:\Om\mllm\data\video_metadata")
OUT_ROOT = Path(r"E:\Om\mllm\data\videomedia")

# yt-dlp throttling / politeness
SLEEP_BETWEEN_VIDEOS_SEC = (0.0, 10.0)   # extra sleep you control
YTDLP_SLEEP_INTERVAL = 0               # yt-dlp sleeps before each download
YTDLP_MAX_SLEEP_INTERVAL = 0
DENO_PATH = r"C:\Users\omras\AppData\Local\Microsoft\WinGet\Packages\DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe\deno.exe"


# Choose your format (keep your current one, or switch to bv*+ba/best for robustness)
YTDLP_FORMAT = "bv*[height<=1080]+ba/best"
# YTDLP_FORMAT = "bv*+ba/best"

COOKIES_FROM_BROWSER = ("firefox",)     # set to None if you don't want this
# COOKIE_FILE = r"E:\Om\mllm\youtube_cookies.txt"  # alternative
# ------------------------------------------------


def sanitize_folder_name(name: str) -> str:
    """Windows-safe-ish folder name."""
    if not name:
        return "unknown_channel"
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    name = name.strip().strip(".")
    return name[:120] if len(name) > 120 else name


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_video_list(payload):
    """
    Supports:
      - list[dict]
      - dict with keys like "videos", "items", "data"
      - dict that itself looks like a single video record
    """
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("videos", "items", "data", "results"):
            v = payload.get(key)
            if isinstance(v, list):
                return v
        # fallback: maybe it *is* a single record
        return [payload]

    return []


def get_video_id(rec: dict) -> str | None:
    for k in ("id", "video_id", "videoId", "videoID", "youtube_id"):
        v = rec.get(k)
        if v:
            return str(v).strip()
    return None


def get_channel_name(file_path: Path, payload, rec: dict) -> str:
    # record-level hints
    for k in ("channel_name", "channelTitle", "channel", "handle", "brand_name"):
        v = rec.get(k)
        if v:
            return str(v).strip()

    # payload-level hints
    if isinstance(payload, dict):
        for k in ("channel_name", "channelTitle", "channel", "handle", "brand_name"):
            v = payload.get(k)
            if v:
                return str(v).strip()

    # filename fallback: "<channel>.dnwd.json" -> "<channel>"
    name = file_path.name
    if name.endswith(".dnwd.json"):
        return name[:-len(".dnwd.json")]
    return file_path.stem


def video_already_exists(out_dir: Path, video_id: str) -> bool:
    # New naming: "<video_id>.<ext>"
    if any(out_dir.glob(f"{video_id}.*")):
        return True
    # Old naming you used: "<duration>_<video_id>.<ext>"
    if any(out_dir.glob(f"*_{video_id}.*")):
        return True
    return False


def download_one(video_id: str, out_dir: Path) -> None:
    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "format": YTDLP_FORMAT,
        "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 8,
        "sleep_interval": YTDLP_SLEEP_INTERVAL,
        "max_sleep_interval": YTDLP_MAX_SLEEP_INTERVAL,
        "js_runtimes": {"deno": {"path": DENO_PATH}},
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    if COOKIES_FROM_BROWSER:
        ydl_opts["cookiesfrombrowser"] = COOKIES_FROM_BROWSER

    # If you prefer cookiefile instead:
    # ydl_opts["cookiefile"] = COOKIE_FILE

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def main():
    files = sorted(METADATA_DIR.glob("*.dnwd.json"))
    if not files:
        print(f"No .dnwd.json files found in: {METADATA_DIR}")
        return

    tasks = []
    for fp in files:
        try:
            payload = load_json(fp)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
            continue

        records = extract_video_list(payload)
        for rec in records:
            if not isinstance(rec, dict):
                continue
            vid = get_video_id(rec)
            if not vid:
                continue
            ch = sanitize_folder_name(get_channel_name(fp, payload, rec))
            tasks.append((ch, vid))

    # de-dupe
    tasks = list(dict.fromkeys(tasks))

    for ch, vid in tqdm(tasks, desc="Downloading", unit="video"):
        out_dir = OUT_ROOT / ch
        out_dir.mkdir(parents=True, exist_ok=True)

        if video_already_exists(out_dir, vid):
            continue
        # download_one(vid, out_dir)
        try:
            download_one(vid, out_dir)
        except Exception as e:
            tqdm.write(f"[ERR] {ch} / {vid}: {e}")

        # extra politeness to avoid YouTube rejecting bursts
        # time.sleep(skewed_sleep())


if __name__ == "__main__":
    main()
