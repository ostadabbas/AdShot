from googleapiclient.discovery import build
import pandas as pd
import tqdm
import isodate

from pytube.innertube import _default_clients
from pytube import cipher
import re
from langdetect import detect, DetectorFactory
# Set seed for reproducibility in langdetect
DetectorFactory.seed = 0
import langid

from googleapiclient.discovery import build
from datetime import datetime
import pandas as pd

_default_clients["ANDROID"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["ANDROID_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_MUSIC"]["context"]["client"]["clientVersion"] = "6.41"
_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]


class YouTubeClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.youtube = self.build_youtube(api_key)
        # Patch pytube throttling function on class initialization
        cipher.get_throttling_function_name = self.get_throttling_function_name

    @staticmethod
    def get_throttling_function_name(js: str) -> str:
        """Extract the name of the function that computes the throttling parameter."""
        function_patterns = [
            r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&\s*'
            r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])?\([a-z]\)',
            r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])\([a-z]\)',
        ]
        for pattern in function_patterns:
            regex = re.compile(pattern)
            function_match = regex.search(js)
            if function_match:
                if len(function_match.groups()) == 1:
                    return function_match.group(1)
                idx = function_match.group(2)
                if idx:
                    idx = idx.strip("[]")
                    array = re.search(
                        r'var {nfunc}\s*=\s*(\[.+?\]);'.format(
                            nfunc=re.escape(function_match.group(1))),
                        js
                    )
                    if array:
                        array = array.group(1).strip("[]").split(",")
                        array = [x.strip() for x in array]
                        return array[int(idx)]

        raise Exception(
            "RegexMatchError: get_throttling_function_name failed to match function name"
        )

    @staticmethod
    def build_youtube(api_key=None):
        return build('youtube', 'v3', developerKey=api_key)

    def get_playlists(self, channel_id):
        playlists = []
        request = self.youtube.playlists().list(
            part="id,snippet",
            channelId=channel_id,
            maxResults=50
        )
        response = request.execute()

        while request is not None:
            for item in response['items']:
                playlists.append({
                    'id': item['id'],
                    'title': item['snippet']['title']
                })
            request = self.youtube.playlists().list_next(request, response)
            if request:
                response = request.execute()

        return playlists

    def get_videos_from_playlist(self, playlist_id):
        videos = []
        request = self.youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=playlist_id,
            maxResults=50
        )
        response = request.execute()

        while request is not None:
            for item in response['items']:
                video_id = item['contentDetails']['videoId']
                video_title = item['snippet']['title']
                videos.append({
                    'id': video_id,
                    'title': video_title
                })
            request = self.youtube.playlistItems().list_next(request, response)
            if request:
                response = request.execute()

        return videos

    def get_video_details(self, video_id):
        request = self.youtube.videos().list(
            part="contentDetails,snippet",
            id=video_id
        )
        response = request.execute()
        if 'items' in response and len(response['items']) > 0:
            video = response['items'][0]
            duration = video['contentDetails']['duration']
            return {
                'id': video_id,
                'title': video['snippet']['title'],
                'duration': duration
            }
        else:
            return None

    @staticmethod
    def parse_duration(duration):
        parsed_duration = isodate.parse_duration(duration)
        return parsed_duration.total_seconds()

    def get_channel_language(self, channel_id):
        youtube = build('youtube', 'v3', developerKey=self.api_key)
        request = youtube.channels().list(
            part="snippet,brandingSettings",
            id=channel_id
        )
        response = request.execute()
        if 'items' in response and len(response['items']) > 0:
            channel = response['items'][0]
            if 'brandingSettings' in channel and 'channel' in channel['brandingSettings']:
                language = channel['brandingSettings']['channel'].get('defaultLanguage', 'en')
                return language
        return 'en'

    def get_channel_id(self, channel_handle):
        resp = self.youtube.search().list(
            q=channel_handle,
            type="channel",
            part="snippet",
            maxResults=1
        ).execute()
        channel_id = resp["items"][0]["id"]["channelId"]
        return channel_id

# USAGE EXAMPLE for handle "motts"
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    yt_client = YouTubeClient(api_key=api_key)
    handle = "@Fanta"
    channel_id = yt_client.get_channel_id(handle)
    playlists = yt_client.get_playlists(channel_id)
    print(playlists)
    # optionally: process playlist videos
    if playlists:
        playlist_id = playlists[0]['id']
        videos = yt_client.get_videos_from_playlist(playlist_id)
        for v in videos:
            details = yt_client.get_video_details(v['id'])
            print(details)
