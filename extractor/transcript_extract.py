from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import time
import random
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound
)

class YouTubeTranscriptExtractor:
    """
    Responsible for extracting transcript + metadata
    """
    def __init__(self, api_key: str, delay_range=(2.5, 3.0)):
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self.transcript_api = YouTubeTranscriptApi()
        self.delay_range = delay_range

    # Fetch video's metadata
    def get_video_info(self, video_id: str) -> dict | None:
        try:
            request = self.youtube.videos().list(
                part="snippet",
                id=video_id
            )
            response = request.execute()

            if not response["items"]:
                return None

            snippet = response["items"][0]["snippet"]

            return {
                "title": snippet["title"],
                "channel": snippet["channelTitle"],
                "published": snippet["publishedAt"]
            }

        except Exception as e:
            print(f"[ERROR] video info {video_id}: {e}")
            return None

    # Fetch video's transcript
    def get_transcript(self, video_id: str) -> list:
        try:
            transcript = self.transcript_api.fetch(video_id)
            return transcript.to_raw_data()

        except TranscriptsDisabled:
            print(f"[WARN] transcripts disabled: {video_id}")
        except NoTranscriptFound:
            print(f"[WARN] no transcript found: {video_id}")
        except Exception as e:
            print(f"[ERROR] transcript {video_id}: {e}")

        return []

    # Return info and transcript for each video
    def extract(self, video_id: str) -> dict | None:
        info = self.get_video_info(video_id)
        if not info:
            return None

        transcript = self.get_transcript(video_id)

        time.sleep(random.uniform(*self.delay_range))

        return {
            "video_id": video_id,
            **info,
            "transcript": transcript
        }
