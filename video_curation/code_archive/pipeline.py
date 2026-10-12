if __name__ == "__main__":
    import json
    from pathlib import Path
    from export_channel_videos import export_channel

    path = Path(r"E:\Om\mllm\data\channels\@heineken.json")
    flag_path = Path(r"E:\Om\mllm\data\channels\flag.json")

    # Load raw channel file
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    print("@heineken.json loaded:")
    print(raw)

    # Load flags
    with flag_path.open("r", encoding="utf-8") as f:
        flags = json.load(f)

    flag_map = {
        cid: bool(v.get("include", True))
        for cid, v in (flags.get("channels", {}) or {}).items()
    }

    def included(ch: dict) -> bool:
        cid = ch.get("channel_id") or ch.get("channelId")
        # default include if id missing or not present in flag.json
        return flag_map.get(cid, True)

    # Filter each list while keeping same JSON structure
    raw["featured_channels"] = [ch for ch in raw.get("featured_channels", []) if included(ch)]
    raw["searched_channels"] = [ch for ch in raw.get("searched_channels", []) if included(ch)]

    # Merge featured_channels and searched_channels
    merged_channels = raw.get("featured_channels", []) + raw.get("searched_channels", [])
    # Print the handle of each merged channel
    # Extract YouTube API key (from environment variable or a config file)
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    for ch in merged_channels:
        handle = ch.get("handle")
        playlist_path, videos_path = export_channel(
            handle, 
            api_key,
            videos_dir = "data/videos/heineken",
            )
        