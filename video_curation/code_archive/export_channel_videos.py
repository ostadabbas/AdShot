from pathlib import Path
from youtube_setup import YouTubeClient  
import tqdm
import isodate
import pandas as pd



def get_videos(channel_id, yt_client):
    videos = []
    next_page_token = None

    while True:
        request = yt_client.youtube.search().list(
            part='id,snippet',
            channelId=channel_id,
            maxResults=50,
            type='video',
            pageToken=next_page_token
        )
        response = request.execute()

        for item in tqdm.tqdm(response['items']):
            video_id = item['id']['videoId']
            video_request = yt_client.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            )
            video_response = video_request.execute()
            video_info = video_response['items'][0]

            video_data = {
                'id': video_id,
                'title': video_info['snippet']['title'],
                'description': video_info['snippet']['description'],
                'channelId': video_info['snippet']['channelId'],
                'channelTitle': video_info['snippet']['channelTitle'],
                'tags': ','.join(video_info['snippet'].get('tags', [])),
                'categoryId': video_info['snippet']['categoryId'],
                'liveBroadcastContent': video_info['snippet']['liveBroadcastContent'],
                'defaultLanguage': video_info['snippet'].get('defaultLanguage', ''),
                'localized_title': video_info['snippet']['localized']['title'],
                'localized_description': video_info['snippet']['localized']['description'],
                'duration': video_info['contentDetails']['duration'],
                'dimension': video_info['contentDetails']['dimension'],
                'definition': video_info['contentDetails']['definition'],
                'caption': video_info['contentDetails']['caption'],
                'licensedContent': video_info['contentDetails']['licensedContent'],
                'projection': video_info['contentDetails']['projection'],
                'viewCount': video_info['statistics'].get('viewCount', 0),
                'likeCount': video_info['statistics'].get('likeCount', 0),
                'dislikeCount': video_info['statistics'].get('dislikeCount', 0),
                'favoriteCount': video_info['statistics'].get('favoriteCount', 0),
                'commentCount': video_info['statistics'].get('commentCount', 0),
            }

            videos.append(video_data)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return videos



def export_channel(
    handle: str,
    api_key: str,
    max_duration: int = 70,
    playlist_dir: str | Path = "data/playlist",
    videos_dir: str | Path = "data/videos",
    playlist_filename: str | None = None,
    videos_filename: str | None = None,
) -> tuple[str, str]:
    """
    Builds:
      1) <playlist_dir>/<playlist_filename>  (playlist-based list)
      2) <videos_dir>/<videos_filename>      (all videos via search)
    Saves as JSON and returns (playlist_json_path, videos_json_path).
    """
    yt_client = YouTubeClient(api_key=api_key)

    playlist_dir = Path(playlist_dir)
    videos_dir = Path(videos_dir)
    playlist_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    safe_handle = handle.lstrip("@").replace("/", "_").replace("\\", "_").replace(" ", "_")
    if playlist_filename is None:
        playlist_filename = f"{safe_handle}.json"
    if videos_filename is None:
        videos_filename = f"{safe_handle}.json"

    # ---- 1) Playlist-based export ----
    channel_id = yt_client.get_channel_id(handle)
    playlists = yt_client.get_playlists(channel_id)

    all_videos = []
    for playlist in tqdm.tqdm(playlists):
        vids = yt_client.get_videos_from_playlist(playlist["id"])
        for video in vids:
            details = yt_client.get_video_details(video["id"])
            if not details:
                continue
            details["playlist_name"] = playlist["title"]
            details["brand_name"] = handle
            details["duration"] = yt_client.parse_duration(details["duration"])  # seconds
            all_videos.append(details)

    if all_videos:
        df = pd.DataFrame(all_videos, columns=["brand_name", "playlist_name", "title", "duration", "id"])
        df.rename(columns={"title": "video_name", "duration": "video_duration", "id": "video_id"}, inplace=True)
        df = df[df["video_duration"] < max_duration].sort_values(
            by=["playlist_name", "video_name", "video_duration"]
        )
        df["url"] = "https://www.youtube.com/watch?v=" + df["video_id"]
    else:
        df = pd.DataFrame(columns=["brand_name", "playlist_name", "video_name", "video_duration", "video_id", "url"])

    playlist_json_path = playlist_dir / playlist_filename
    playlist_json_path.write_text(df.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")

    # ---- 2) Raw channel videos export (search-based) ----
    videos = get_videos(channel_id, yt_client)
    if videos:
        df_v = pd.DataFrame(videos)
        df_v["video_duration"] = df_v["duration"].apply(yt_client.parse_duration)  # seconds
        df_v = df_v[df_v["video_duration"] < max_duration].sort_values(by=["title", "video_duration"])
        df_v["url"] = "https://www.youtube.com/watch?v=" + df_v["id"]
        df_v["handle"] = handle

        ordered_columns = ["handle", "title", "video_duration", "id", "url"] + [
            c for c in df_v.columns if c not in ["handle", "title", "video_duration", "id", "url"]
        ]
        df_v = df_v[ordered_columns]
    else:
        df_v = pd.DataFrame(columns=["handle", "title", "video_duration", "id", "url"])

    videos_json_path = videos_dir / videos_filename
    videos_json_path.write_text(df_v.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")

    return str(playlist_json_path), str(videos_json_path)

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    yt_client = YouTubeClient(api_key=api_key)
    handle = "@hieneken"
    playlist_path, videos_path = export_channel(handle, api_key)

