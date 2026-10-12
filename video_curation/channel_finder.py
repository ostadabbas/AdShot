import json
import os
import re
from collections import deque, defaultdict
from pathlib import Path

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

CHANNEL_ID_RE = re.compile(r"^UC[\w-]{22}$")

# ========== Filesystem-safe filename sanitization ==========
def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be safe for filesystem (esp on Windows):
    - Removes/replace forbidden characters: < > : " / \ | ? *
    - Also strips trailing dots and spaces (not allowed on Windows)
    """
    forbidden = r'[<>:"/\\|?*\n\r\t]'
    cleaned = re.sub(forbidden, "_", name)
    cleaned = cleaned.rstrip(". ").strip()
    # Limit filename length to 128 to avoid overly long paths
    return cleaned[:128]

def _yt(api_key: str):
    return build("youtube", "v3", developerKey=api_key)

def _chunks(xs, n=50):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def resolve_channel_id(api_key: str, seed: str) -> str | None:
    yt = _yt(api_key)
    seed = (seed or "").strip()

    # 1) Already a channelId?
    if CHANNEL_ID_RE.match(seed):
        try:
            r = yt.channels().list(part="id", id=seed).execute()
            return r["items"][0]["id"] if r.get("items") else None
        except Exception:
            return None

    # 2) Handle-like: try forHandle, but DO NOT rely on it
    if seed.startswith("@") and " " not in seed:
        # try forHandle first (if supported)
        try:
            r = yt.channels().list(part="id", forHandle=seed).execute()
            if r.get("items"):
                return r["items"][0]["id"]
        except Exception:
            pass  # fall through to search

        # search fallback (handles often resolve fine via search)
        q = seed[1:]
    else:
        q = seed

    # 3) Search fallback (robust parsing: id.channelId)
    try:
        s = yt.search().list(part="id,snippet", q=q, type="channel", maxResults=5).execute()
        for item in s.get("items", []):
            cid = (item.get("id") or {}).get("channelId")
            if cid:
                return cid
    except Exception:
        return None

    return None

def get_featured_channel_ids(api_key: str, channel_id: str) -> list[str]:
    yt = _yt(api_key)
    resp = yt.channelSections().list(
        part="snippet,contentDetails",
        channelId=channel_id
    ).execute()

    featured = []
    for sec in resp.get("items", []):
        sec_type = (sec.get("snippet", {}).get("type") or "")
        norm = sec_type.replace("_", "").lower()
        if norm == "multiplechannels":
            featured.extend(sec.get("contentDetails", {}).get("channels", []) or [])

    return list(dict.fromkeys(featured))

def get_channels_meta_with_handle(api_key: str, channel_ids: list[str]) -> dict:
    """
    Returns {channelId: {"title":..., "customUrl":..., "handle":...}}
    handle = customUrl if it's already like '@xyz', else None.
    (This is the only reliable "handle-like" field you can get from Data API v3.)
    """
    yt = _yt(api_key)
    meta = {}

    ids = [c for c in channel_ids if c]
    for chunk in _chunks(ids, 50):
        r = yt.channels().list(part="snippet", id=",".join(chunk)).execute()
        for it in r.get("items", []):
            cid = it["id"]
            sn = it.get("snippet", {}) or {}
            custom = sn.get("customUrl")
            handle = custom if isinstance(custom, str) and custom.startswith("@") else None
            meta[cid] = {
                "title": sn.get("title"),
                "customUrl": custom,
                "handle": handle,
                "description": sn.get("description"),
            }
    return meta

def bfs_featured_graph(api_key: str, seeds, max_depth: int = 2, max_nodes: int = 300, strict: bool = False):
    if isinstance(seeds, str):
        seeds = [seeds]

    seed_ids = []
    unresolved = []

    for s in seeds:
        cid = resolve_channel_id(api_key, s)
        if cid:
            seed_ids.append(cid)
        else:
            unresolved.append(s)

    if not seed_ids:
        if strict:
            raise ValueError(f"No valid seed channels resolved from: {seeds}")
        # return “empty graph” instead of error
        return {}, {}, {}, {}, unresolved

    visited = set(seed_ids)
    depth_of = {cid: 0 for cid in seed_ids}
    q = deque((cid, 0) for cid in seed_ids)

    edges = defaultdict(list)
    new_by_depth = defaultdict(list)

    while q and len(visited) < max_nodes:
        cur, d = q.popleft()
        if d >= max_depth:
            continue

        try:
            children = get_featured_channel_ids(api_key, cur)
        except HttpError:
            continue

        edges[cur] = children
        nd = d + 1

        for ch in children:
            if ch not in visited:
                visited.add(ch)
                depth_of[ch] = nd
                new_by_depth[nd].append(ch)
                q.append((ch, nd))
                if len(visited) >= max_nodes:
                    break

    nodes_meta = get_channels_meta_with_handle(api_key, list(visited))
    return nodes_meta, dict(edges), dict(new_by_depth), depth_of, unresolved

def search_company_channels(api_key: str, company: str, max_results: int = 25):
    youtube = _yt(api_key)

    # 1) Search YouTube for channels related to the company
    search_resp = youtube.search().list(
        part="snippet",
        q=company,
        type="channel",
        maxResults=max_results
    ).execute()

    channel_ids = [item["id"]["channelId"] for item in search_resp.get("items", [])]
    if not channel_ids:
        return pd.DataFrame()

    # 2) Fetch channel details (title, description, etc.)
    details_resp = youtube.channels().list(
        part="snippet",
        id=",".join(channel_ids)
    ).execute()

    rows = []
    for ch in details_resp.get("items", []):
        sn = ch["snippet"]
        rows.append({
            "channelId": ch["id"],
            "title": sn.get("title"),
            "description": sn.get("description"),
            "customUrl": sn.get("customUrl"),
            "publishedAt": sn.get("publishedAt"),
        })

    return pd.DataFrame(rows)

def generate_channels_raw(
    api_key: str,
    featured_seed="@CocaCola",
    max_depth=3,
    max_nodes=500,
    output_dir="out",
    output_filename="channels_raw.json"
):
    """
    Generate channels_raw.json given a YouTube API key and search parameters.
    Handles special cases such as brands with slashes or illegal characters in their names.
    """
    # Sanitize the output filename to avoid file system errors (esp. for Windows with ?, :, /, etc)
    # If the filename looks unsafe, apply sanitization
    safe_output_filename = sanitize_filename(output_filename)

    nodes, edges, new_by_depth, depth_of, unresolved = bfs_featured_graph(
        api_key,
        seeds=featured_seed,
        max_depth=max_depth,
        max_nodes=max_nodes
    )

    channels = search_company_channels(api_key, featured_seed)
    searched_channels = [
        {
            "title": row.get("title", ""),
            "channel_id": row.get("channelId", ""),
            "handle": row.get("customUrl", ""),
            "description": row.get("description", ""),
        }
        for _, row in channels.iterrows()
    ]

    featured_channels = [
        {
            "title": v.get("title", ""),
            "channel_id": k,
            "handle": v.get("handle", ""),
            "description": v.get("description", ""),
        }
        for k, v in nodes.items()
    ]
    
    # Remove overlap: filter searched_channels where channel_id not in featured_channels set
    featured_ids = {ch["channel_id"] for ch in featured_channels}
    searched_channels_no_overlap = [
        ch for ch in searched_channels if ch.get("channel_id") not in featured_ids
    ]
    out = {
        "featured_channels": featured_channels,
        "searched_channels": searched_channels_no_overlap,
    }
    out_path = Path(output_dir) / safe_output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path

def export_featured_channels(
    api_key: str,
    channel_id: str,                      # PASS UC... to avoid search.list entirely
    output_dir: str = "out",
    output_filename: str = "featured_channels.json",
    with_meta: bool = True,               # set False for absolute minimum quota
):
    """
    Exports the channels featured on ONE channel's homepage (multipleChannels sections).

    Requires:
      - get_featured_channel_ids(api_key, channel_id)
      - get_channels_meta_with_handle(api_key, channel_ids)  (only if with_meta=True)
      - sanitize_filename(name)
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / sanitize_filename(output_filename)

    featured_ids = get_featured_channel_ids(api_key, channel_id)  # <-- ONLY channelSections.list

    out = {
        "seed_channel_id": channel_id,
        "featured_channel_ids": featured_ids,
        "featured_count": len(featured_ids),
    }

    if with_meta and featured_ids:
        meta = get_channels_meta_with_handle(api_key, featured_ids)  # <-- channels.list in chunks of 50
        out["featured_channels"] = [
            {
                "channel_id": cid,
                "title": (meta.get(cid) or {}).get("title", ""),
                "handle": (meta.get(cid) or {}).get("handle", ""),
                "customUrl": (meta.get(cid) or {}).get("customUrl", ""),
                "description": (meta.get(cid) or {}).get("description", ""),
            }
            for cid in featured_ids
        ]

    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    channel_seed = "Macy's"
    generate_channels_raw(
        os.getenv("YOUTUBE_API_KEY"),
        featured_seed=channel_seed,
        output_dir="data/channels",
        output_filename=f"{channel_seed}.json",
    )
