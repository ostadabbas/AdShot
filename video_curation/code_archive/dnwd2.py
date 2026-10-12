from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import isodate
import yt_dlp
from tqdm import tqdm


@dataclass(frozen=True)
class VideoJob:
    seed: str
    video_id: str
    duration_s: int
    source_json: Path


def safe_seed(seed: str) -> str:
    s = (seed or "unknown").strip()
    s = s.lstrip("@")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return (s[:80] or "unknown")


def parse_duration_seconds(iso_8601: str) -> Optional[int]:
    if not iso_8601:
        return None
    try:
        td = isodate.parse_duration(iso_8601)
        return int(td.total_seconds())
    except Exception:
        return None


def iter_json_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.json")


def load_jobs_from_json(json_path: Path, max_seconds: int) -> list[VideoJob]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    # Normalize to a list of "records" (each record has a seed + videos list)
    records = []

    if isinstance(data, dict):
        records = [data]

    elif isinstance(data, list):
        # Case A: list of channel-record dicts (each may contain "videos")
        if data and all(isinstance(x, dict) for x in data) and any("videos" in x for x in data):
            records = data
        # Case B: list of video dicts directly (each has "id", "duration", etc.)
        elif data and all(isinstance(x, dict) for x in data) and any("id" in x for x in data):
            records = [{"seed": json_path.stem, "videos": data}]
        else:
            return []

    else:
        return []

    jobs: list[VideoJob] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue

        seed = rec.get("seed") or rec.get("channelTitle") or rec.get("channelId") or json_path.stem
        videos = rec.get("videos") or []

        if not isinstance(videos, list):
            continue

        for v in videos:
            if not isinstance(v, dict):
                continue

            vid = str(v.get("id") or "").strip()
            dur_iso = v.get("duration")  # e.g. "PT30S"
            dur_s = parse_duration_seconds(dur_iso)

            if not vid or dur_s is None:
                continue
            if dur_s < max_seconds:
                jobs.append(
                    VideoJob(
                        seed=str(seed),
                        video_id=vid,
                        duration_s=dur_s,
                        source_json=json_path
                    )
                )

    return jobs


def already_downloaded(dest_dir: Path, video_id: str) -> bool:
    # Any file containing _<video_id>.<ext> counts as present
    return any(dest_dir.glob(f"*_{video_id}.*"))


def download_one(
    job: VideoJob,
    out_dir: Path,
    *,
    format_spec: str,
    cookiesfrombrowser: Optional[tuple],
    cookiefile: Optional[Path],
    concurrent_fragments: int,
) -> tuple[str, str]:
    """
    Returns (video_id, status) where status is one of:
      - "ok"
      - "skipped_exists"
      - "failed:<reason>"
    """
    seed_dir = out_dir / safe_seed(job.seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    if already_downloaded(seed_dir, job.video_id):
        return (job.video_id, "skipped_exists")

    url = f"https://www.youtube.com/watch?v={job.video_id}"
    outtmpl = str(seed_dir / f"{job.duration_s:03d}s_{job.video_id}.%(ext)s")

    ydl_opts = {
        "format": format_spec,
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": int(concurrent_fragments),
        "continuedl": True,
        "quiet": True,
        "no_warnings": True,
    }

    if cookiefile:
        ydl_opts["cookiefile"] = str(cookiefile)
    elif cookiesfrombrowser:
        ydl_opts["cookiesfrombrowser"] = cookiesfrombrowser

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return (job.video_id, "ok")
    except Exception as e:
        return (job.video_id, f"failed:{type(e).__name__}:{e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", type=str, default=r"E:\Om\mllm\data\videos")
    ap.add_argument("--out-dir", type=str, default=r"E:\Om\mllm\data\videomedia")
    ap.add_argument("--max-seconds", type=int, default=70)
    ap.add_argument("--workers", type=int, default=4)

    # Prefer robust format; your current script uses 137+140/best :contentReference[oaicite:2]{index=2}
    ap.add_argument(
        "--format",
        type=str,
        default="bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/best",
        help="yt-dlp format selector",
    )

    # Cookies: default matches your current approach (Firefox) :contentReference[oaicite:3]{index=3}
    ap.add_argument("--cookies-browser", type=str, default="firefox")
    ap.add_argument("--cookies-profile", type=str, default=None, help="optional browser profile name/path")
    ap.add_argument("--cookiefile", type=str, default=None, help="path to exported cookies.txt (optional)")

    ap.add_argument("--concurrent-fragments", type=int, default=4)

    args = ap.parse_args()

    json_root = Path(args.json_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Collect jobs
    jobs: list[VideoJob] = []
    for jp in iter_json_files(json_root):
        jobs.extend(load_jobs_from_json(jp, args.max_seconds))

    # 2) Deduplicate by video_id (keep first occurrence)
    seen = set()
    deduped: list[VideoJob] = []
    for j in jobs:
        if j.video_id in seen:
            continue
        seen.add(j.video_id)
        deduped.append(j)

    if not deduped:
        print("No videos found under threshold.")
        return

    # Cookies tuple for yt-dlp
    cookiesfrombrowser = None
    if args.cookiefile:
        cookiefile = Path(args.cookiefile)
    else:
        cookiefile = None
        # yt-dlp supports ('firefox',) or ('firefox', 'profile')
        if args.cookies_profile:
            cookiesfrombrowser = (args.cookies_browser, args.cookies_profile)
        else:
            cookiesfrombrowser = (args.cookies_browser,)

    # 3) Download
    ok = skipped = failed = 0
    failures = []

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [
            ex.submit(
                download_one,
                j,
                out_dir,
                format_spec=args.format,
                cookiesfrombrowser=cookiesfrombrowser,
                cookiefile=cookiefile,
                concurrent_fragments=args.concurrent_fragments,
            )
            for j in deduped
        ]

        for f in tqdm(as_completed(futs), total=len(futs), desc="Downloading"):
            vid, status = f.result()
            if status == "ok":
                ok += 1
            elif status == "skipped_exists":
                skipped += 1
            else:
                failed += 1
                failures.append((vid, status))

    print(f"\nDone. ok={ok}, skipped_exists={skipped}, failed={failed}")
    if failures:
        fail_path = out_dir / "_failures.txt"
        fail_path.write_text("\n".join([f"{vid}\t{st}" for vid, st in failures]), encoding="utf-8")
        print(f"Failures saved to: {fail_path}")


if __name__ == "__main__":
    main()
