import os
import json
from typing import Optional, List, Dict

import isodate
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

def build_yt(api_key: str):
    return build("youtube", "v3", developerKey=api_key)

def resolve_channel_id(api_key: str, seed: str) -> Optional[str]:
    """
    Accepts:
      - UC... channelId
      - @handle
      - https://www.youtube.com/@handle
      - https://www.youtube.com/channel/UC...
      - plain string -> fallback search (expensive)
    """
    yt = build_yt(api_key)
    s = seed.strip()

    # 1) Direct channelId
    if s.startswith("UC") and len(s) >= 20:
        return s

    # 2) URL forms
    if "youtube.com/channel/" in s:
        try:
            return s.split("youtube.com/channel/")[1].split("/")[0].split("?")[0]
        except Exception:
            pass

    # 3) Handle forms
    if "youtube.com/@" in s:
        s = "@" + s.split("youtube.com/@", 1)[1].split("/")[0].split("?")[0]
    if s.startswith("@"):
        handle = s[1:]
        resp = yt.channels().list(part="id", forHandle=handle).execute()
        items = resp.get("items", [])
        return items[0]["id"] if items else None

    # 4) Fallback: search (COSTLY: 100 units per call)
    resp = yt.search().list(part="snippet", q=s, type="channel", maxResults=1).execute()
    items = resp.get("items", [])
    if not items:
        return None
    return items[0]["id"]["channelId"]

def get_uploads_playlist_id(api_key: str, channel_id: str) -> Optional[str]:
    yt = build_yt(api_key)
    resp = yt.channels().list(part="contentDetails", id=channel_id).execute()
    items = resp.get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

def list_all_upload_video_ids(api_key: str, uploads_playlist_id: str) -> List[str]:
    yt = build_yt(api_key)
    ids: List[str] = []
    token = None

    while True:
        resp = yt.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=token
        ).execute()

        for it in resp.get("items", []):
            ids.append(it["contentDetails"]["videoId"])

        token = resp.get("nextPageToken")
        if not token:
            break

    return ids

def fetch_video_details(api_key: str, video_ids: List[str]) -> List[Dict]:
    yt = build_yt(api_key)
    out: List[Dict] = []

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        resp = yt.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch)
        ).execute()

        for v in resp.get("items", []):
            sn = v.get("snippet", {}) or {}
            cd = v.get("contentDetails", {}) or {}
            st = v.get("statistics", {}) or {}

            out.append({
                "id": v.get("id"),
                "title": sn.get("title", ""),
                "description": sn.get("description", ""),
                "publishedAt": sn.get("publishedAt", ""),
                "channelId": sn.get("channelId", ""),
                "channelTitle": sn.get("channelTitle", ""),
                "tags": sn.get("tags", []),
                "categoryId": sn.get("categoryId", ""),
                "defaultLanguage": sn.get("defaultLanguage", ""),
                "duration": cd.get("duration", ""),
                "definition": cd.get("definition", ""),
                "caption": cd.get("caption", ""),
                "viewCount": st.get("viewCount", 0),
                "likeCount": st.get("likeCount", 0),
                "commentCount": st.get("commentCount", 0),
            })

    return out

def parse_duration_seconds(duration_iso):
    """Parse ISO 8601 duration to seconds (int), else None."""
    try:
        return int(isodate.parse_duration(duration_iso).total_seconds())
    except Exception:
        return None

def export_channel_uploads(seed: str, out_json: str = "uploads_videos.json"):
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing YOUTUBE_API_KEY in environment/.env")

    try:
        print("Seed", seed)
        channel_id = resolve_channel_id(api_key, seed)
        if not channel_id:
            raise ValueError(f"Could not resolve channel from seed: {seed}")

        uploads_pid = get_uploads_playlist_id(api_key, channel_id)
        if not uploads_pid:
            raise ValueError(f"Could not get uploads playlist for channelId: {channel_id}")

        video_ids = list_all_upload_video_ids(api_key, uploads_pid)
        videos = fetch_video_details(api_key, video_ids)

        # Add exact video_duration (in seconds) field
        for v in videos:
            v["video_duration"] = parse_duration_seconds(v.get("duration"))

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": seed,
                    "channelId": channel_id,
                    "uploadsPlaylistId": uploads_pid,
                    "videoCount": len(videos),
                    "videos": videos,
                },
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"Saved {len(videos)} videos to {out_json}")

    except HttpError as e:
        # show clean error
        print("YouTube API error:", e)

if __name__ == "__main__":
    export_channel_uploads("@heineken", "heineken_uploads.json")
