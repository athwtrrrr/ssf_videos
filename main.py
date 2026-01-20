import os
import json
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from googleapiclient.discovery import build
from dotenv import load_dotenv
import time
import random

# -------------------------------
# CONFIGURATION
# -------------------------------
 
# YouTube Data API key
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# List of video IDs to process
video_ids = [
    "nl7IsRVOWBQ",
    "0RZczdJeTZ0",
    "nLh3AjbPAL0",
    "Ul35-p7riEo",
    "dXXRD3zxa6Y",
    "SV-YLB6vlHk",
    "BwCrXUFx5uc",
    "BvY9P_-wHZM",
    "LaWJKAU9dJU",
    "l_nF0DF-v1o",
    "US18660Z_4c"
    ]

# Folder to store merged JSON files
TRANSCRIPTS_DIR = "transcripts"
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# -------------------------------
# INITIALIZE API CLIENTS
# -------------------------------

# YouTube Data API client
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# YouTube Transcript API
ytt_api = YouTubeTranscriptApi()

# -------------------------------
# PROCESS EACH VIDEO
# -------------------------------

for vid in video_ids:
    print(f"\nProcessing video: {vid}")

    # Fetch video info
    try:
        request = youtube.videos().list(part="snippet", id=vid)
        response = request.execute()

        if not response["items"]:
            print(f"No video info found for {vid}")
            continue

        snippet = response["items"][0]["snippet"]
        title = snippet["title"]
        channel = snippet["channelTitle"]
        published = snippet["publishedAt"]

    except Exception as e:
        print(f"Error fetching video info for {vid}: {e}")
        continue

    # Fetch transcript
    try:
        transcript = ytt_api.fetch(vid)
        raw_transcript = transcript.to_raw_data()
    except TranscriptsDisabled:
        print(f"Transcripts disabled for {vid}")
        raw_transcript = []
    except NoTranscriptFound:
        print(f"No transcript found for {vid}")
        raw_transcript = []
    except Exception as e:
        print(f"Error fetching transcript for {vid}: {e}")
        raw_transcript = []

    # Merge info + transcript
    merged_data = {
        "video_id": vid,
        "title": title,
        "channel": channel,
        "published": published,
        "transcript": raw_transcript
    }

    # Save to JSON file
    file_path = os.path.join(TRANSCRIPTS_DIR, f"{vid}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Saved merged JSON: {file_path}")

    time.sleep(random.uniform(2.5, 3))


