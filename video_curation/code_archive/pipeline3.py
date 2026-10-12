def process_first_featured_and_search_channel(raw, api_key, out_dir):
    from export_channel2 import export_channel_uploads  # Use export_channel_uploads from export_channel2.py
    featured = raw.get("featured_channels", [])
    searched = raw.get("searched_channels", [])

    first_featured = featured[0] if featured else None

    # If the handle for the first featured is None, fall back to first searched
    if first_featured and first_featured.get("handle") is not None:
        channels_to_process = first_featured
    else:
        first_searched = searched[0] if searched else None
        channels_to_process = first_searched
    for ch in [channels_to_process]:   
        handle = ch.get("handle")
        if handle is None:
            continue
        # The export_channel_uploads function takes: handle, out_json (output filename)
        safe_handle = (handle or "unknown").lstrip("@").replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_json = f"{out_dir}/{safe_handle}_uploads.json"
        export_channel_uploads(
            handle,
            out_json=out_json,
        )

if __name__ == "__main__":
    import json
    from pathlib import Path
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")

    channels_dir = Path(r"E:\Om\mllm\data\channels")

    for path in channels_dir.glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        print(f"{path.name} loaded:")

        # Use the stem (filename without extension) as the output subdir for uploads
        stem = path.stem
        uploads_out_dir = f"data/videos/{stem}"
        uploads_dir_path = Path(uploads_out_dir)
        uploads_dir_path.mkdir(parents=True, exist_ok=True)

        # Determine what handles would be processed by process_first_featured_and_search_channel
        featured = raw.get("featured_channels", [])
        searched = raw.get("searched_channels", [])
        first_featured = featured[0] if featured else None
        first_searched = searched[0] if searched else None
        channels_to_process = [ch for ch in [first_featured, first_searched] if ch is not None]

        skip = True
        for ch in channels_to_process:
            handle = ch.get("handle")
            print(f"Processing handle: {handle}")
            safe_handle = (handle or "unknown").lstrip("@").replace("/", "_").replace("\\", "_").replace(" ", "_")
            out_json = uploads_dir_path / f"{safe_handle}_uploads.json"
            if not out_json.is_file():
                skip = False
                break
            # Try loading and validating the JSON
            try:
                with open(out_json, "r", encoding="utf-8") as test_f:
                    _ = json.load(test_f)
            except Exception:
                skip = False
                break

        if skip:
            print(f"Uploads already exist for all relevant handles in {uploads_out_dir}; skipping.")
            continue
        
        print(handle)
        process_first_featured_and_search_channel(
            raw,
            api_key,
            out_dir=uploads_out_dir,
        )