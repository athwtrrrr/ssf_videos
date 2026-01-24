import os
import json
from extractor.transcript_extract import YouTubeTranscriptExtractor
from config import YOUTUBE_API_KEY, RAW_TRANSCRIPTS_DIR

os.makedirs(RAW_TRANSCRIPTS_DIR, exist_ok=True)

video_ids = [
    #Paste video transcript here
    "UJbf_6giwk4"
]

extractor = YouTubeTranscriptExtractor(YOUTUBE_API_KEY)

for vid in video_ids:
    print(f"\nExtracting {vid}")

    data = extractor.extract(vid)

    if not data:
        print(f"Skipping {vid}")
        continue

    path = os.path.join(RAW_TRANSCRIPTS_DIR, f"{vid}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved → {path}")
