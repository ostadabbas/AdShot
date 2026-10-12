import yt_dlp

# Video URL
video_url = "https://www.youtube.com/watch?v=rexd7mc-4jc"
DENO_PATH = r"C:\Users\omras\AppData\Local\Microsoft\WinGet\Packages\DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe\deno.exe"


# Options for downloading
# Correct cookiesfrombrowser must be a tuple, not a string!
# The correct form: ("firefox",) or ("firefox", "profile") if you want a custom profile
ydl_opts = {
    'format': 'bestvideo+bestaudio/best',  # Download best video and audio, then merge
    'outtmpl': '%(title)s.%(ext)s',        # Output file name format
    "cookiesfrombrowser": ("firefox",),    # Correct tuple format!
    "js_runtimes": {"deno": {"path": DENO_PATH}},
    "remote_components": ["ejs:github"],    # key fix (enables EJS scripts)
    "verbose": True,

}

# Download the video
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

print("Download complete!")

# # import yt_dlp

# # video_url = "https://www.youtube.com/watch?v=gnA0dSLRJnY"

# # # Path to your exported cookies file
# # COOKIE_FILE = r"E:\Om\mllm\youtube_cookies.txt"

# # ydl_opts = {
# #     # Use your cookies so age-restricted video works
# #     "cookiefile": COOKIE_FILE,

# #     # Choose a sane default format
# #     "format": "bv*+ba/best",

# #     # Tell yt-dlp which JS runtime to use (Node in this example)
# #     # If you chose Deno instead, use {"deno": {}} here.
# #     "js_runtimes": {"node": {}},

# #     # Optional: show progress
# #     "progress_hooks": [lambda d: print(d["status"], d.get("downloaded_bytes", 0)) if d["status"] in ("downloading", "finished") else None],
# # }

# # if __name__ == "__main__":
# #     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
# #         ydl.download([video_url])

# def download_video_fail(row):
#     print(row)
#     # pair_id = row['pair_id']
#     duration = row['video_duration']
#     video_id = row['id']
#     url = row['url'] 
#     ydl_opts = {
#         'format': '137+140/best',
#         'outtmpl': f'videomedia/{duration}_{video_id}.%(ext)s',
#     }
#     print(url)
    
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         try:
#             ydl.download(url)
#             print(f"Successfully downloaded: {video_id}")
#         except Exception as e:
#             print(f"Error downloading {video_id}: {str(e)}")
            

# if __name__ == "__main__":
#     import pandas as pd

#     # Path to Excel file
#     excel_path = r"E:\Om\mllm\videos\@hieneken_filtered.xlsx"

#     # Read rows from the Excel file
#     df = pd.read_excel(excel_path)

#     # Apply the download_video_fail function to each row
#     for _, row in df.iterrows():
#         download_video_fail(row)

# import yt_dlp
# import pandas as pd

# def download_video(row):
#     duration = row['video_duration']
#     video_id = str(row['id'])


#     # If your Excel has full URLs in a column named 'url':
#     # video_url = str(row['url']).strip()

#     # If your Excel only has video IDs and NOT full URLs:
#     video_url = f"https://www.youtube.com/watch?v={video_id}"

#     ydl_opts = {
#         # "cookiefile": r"E:\Om\mllm\youtube_cookies.txt",
#         'format': '137+140/best',
#         'outtmpl': f'videomedia/{duration}_{video_id}.%(ext)s',
#         'cookiesfrombrowser': ('firefox',),
#     }

#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([video_url])  # NOTE: list + ensure string
#         print(f"Successfully downloaded: {video_id}")
#     except Exception as e:
#         print(f"Error downloading {video_id}: {e}")

# if __name__ == "__main__":
#     excel_path = r"E:\Om\mllm\videos\@hieneken_filtered.xlsx"
#     df = pd.read_excel(excel_path)

#     # Optional: filter out rows where URL/ID is missing
#     df = df.dropna(subset=['id'])

#     for _, row in df.iterrows():
#         download_video(row)
