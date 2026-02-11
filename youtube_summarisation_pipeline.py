# youtube_summarization_pipeline.py
import os
import re
import json
import time
import random
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv
from google import genai  

# ===================== CONFIGURATION =====================
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Directories 
RAW_TRANSCRIPTS_DIR = "data/transcripts"
CLEAN_DIR = "data/cleaned"
GEMINI_SUMMARY_DIR = "data/final_w_gemini"

def init_directories():
    """Create all necessary directories."""
    for dir_path in [RAW_TRANSCRIPTS_DIR, CLEAN_DIR, GEMINI_SUMMARY_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    print("All directories created")

# ===================== EXTRACTION MODULE =====================
class YouTubeTranscriptExtractor:
    """Extract transcript + metadata from YouTube."""
    def __init__(self, api_key: str, delay_range=(2.5, 3.0)):
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self.transcript_api = YouTubeTranscriptApi()
        self.delay_range = delay_range

    def get_video_info(self, video_id: str) -> Optional[Dict]:
        try:
            request = self.youtube.videos().list(part="snippet", id=video_id)
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

    def get_transcript(self, video_id: str) -> List[Dict]:
        try:
            transcript = self.transcript_api.fetch(video_id)
            return transcript.to_raw_data()
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            print(f"[WARN] transcript issue: {video_id} – {e}")
        except Exception as e:
            print(f"[ERROR] transcript {video_id}: {e}")
        return []

    def extract(self, video_id: str) -> Optional[Dict]:
        info = self.get_video_info(video_id)
        if not info:
            return None
        transcript = self.get_transcript(video_id)
        time.sleep(random.uniform(*self.delay_range))
        return {
            "video_id": video_id,
            **info,
            "transcript": transcript,
            "word_count": sum(len(seg["text"].split()) for seg in transcript)
        }

# ===================== CLEANING MODULE =====================
class TranscriptCleaner:
    """Clean filler words, brackets, duplicate words, and fix spacing."""
    def __init__(self):
        self.filler_patterns = [
            r"\bum+\b",
            r"\buh+\b",
            r"\[music\]",
            r"\[applause\]",
            r"\[laughter\]",
            r"\[.*?\]"  
        ]
    
    def clean_text(self, text: str) -> str:
        for pattern in self.filler_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        # remove duplicate consecutive words
        text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return self.capitalize_sentences(text)

    def capitalize_sentences(self, text: str) -> str:
        sentences = re.split(r'([.!?])', text)
        cleaned = []
        for i in range(0, len(sentences)-1, 2):
            sentence = sentences[i].strip()
            punct = sentences[i+1]
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
                cleaned.append(sentence + punct)
        return ' '.join(cleaned)

# ===================== GEMINI SUMMARIZER (NEW CLIENT) =====================
class GeminiSummarizer:
    """Direct summarization for short videos, chunked hierarchical summarization for long ones."""
    
    def __init__(self, api_key: str, 
                 model_name: str = "gemini-2.5-pro",   
                 max_retries: int = 3,
                 long_video_threshold: int = 15000,       
                 chunk_size: int = 5000):                 
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.long_video_threshold = long_video_threshold
        self.chunk_size = chunk_size
        print(f"Using {model_name} ")

    # ------------------------------------------------------------
    # 1.  Short videos – direct summarization 
    # ------------------------------------------------------------
    def _summarize_direct(self, transcript_text: str, metadata: dict) -> str:
        """Single API call – for videos under threshold."""
        prompt = self._create_direct_prompt(transcript_text, metadata)
        return self._call_gemini_with_retry(prompt)

    def _create_direct_prompt(self, transcript_text: str, metadata: dict) -> str:
        return f"""Summarize the following YouTube video transcript precisely

VIDEO TITLE: {metadata.get('title', 'Unknown')}
CHANNEL: {metadata.get('channel', 'Unknown')}

INSTRUCTIONS:
- Do NOT include any introductory phrases.
- Start the summary **immediately** with the first substantive sentence.
- Write a comprehensive summary of 10–15 sentences covering all main topics, key points, examples, and conclusions.
- Use clear, factual, neutral language.

TRANSCRIPT:
{transcript_text}

SUMMARY (start directly):"""

    # ------------------------------------------------------------
    # 2.  Long videos – chunked + hierarchical summarization
    # ------------------------------------------------------------
    def _summarize_long(self, transcript_text: str, metadata: dict) -> str:
        """Two‑stage summarisation: chunk → combine → final summary."""
        print(f"  Long video detected ({len(transcript_text.split()):,} words). Using chunked summarization...")
        
        # ---- Step 1: Split into chunks ----
        words = transcript_text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        print(f"     Split into {len(chunks)} chunks (≈{self.chunk_size} words each)")

        # ---- Step 2: Summarise each chunk ----
        chunk_summaries = []
        for idx, chunk in enumerate(chunks, 1):
            print(f"     Summarizing chunk {idx}/{len(chunks)}...")
            prompt = f"""Summarize this section of a video transcript in 3–5 sentences. 
Focus on the main points, key arguments, and important details.

SECTION:
{chunk}

SUMMARY:"""
            summary = self._call_gemini_with_retry(prompt)
            chunk_summaries.append(summary)
            time.sleep(0.5)  # small delay to avoid hitting rate limits

        # ---- Step 3: Combine chunk summaries ----
        combined = "\n\n".join(chunk_summaries)
        print(f"     Combined {len(chunk_summaries)} chunk summaries ({len(combined.split()):,} words)")

        # ---- Step 4: Final comprehensive summary ----
        final_prompt = f"""Combine the following partial summaries into one coherent, comprehensive summary of the entire video.

VIDEO TITLE: {metadata.get('title', 'Unknown')}
CHANNEL: {metadata.get('channel', 'Unknown')}

PARTIAL SUMMARIES (each covers one segment of the video):
{combined}

INSTRUCTIONS:
- Synthesise the information into a single, well‑structured summary.
- Cover all major themes and key points from the entire video.
- Write 15–20 sentences (longer than a short‑video summary).
- Do NOT include introductory phrases.
- Start the summary immediately.

FULL VIDEO SUMMARY:"""
        
        print("     Generating final combined summary...")
        final_summary = self._call_gemini_with_retry(final_prompt)
        return final_summary

    # ------------------------------------------------------------
    # 3.  Core API caller with retry logic 
    # ------------------------------------------------------------
    def _call_gemini_with_retry(self, prompt: str) -> str:
        """Send prompt to Gemini with exponential backoff retry."""
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"      API call (attempt {attempt}/{self.max_retries})...")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "temperature": 0.3,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 3000,  
                    }
                )
                if not response.text:
                    raise ValueError("Empty response")
                summary = response.text.strip()
                # Minimal cleaning
                prefixes = ["Here is a summary:", "Summary:", "Of course!", "Sure,", "Certainly:"]
                for p in prefixes:
                    if summary.startswith(p):
                        summary = summary[len(p):].lstrip()
                return summary
            except Exception as e:
                error_msg = str(e)
                print(f"      ⚠ Attempt {attempt} failed: {error_msg[:100]}")
                is_retryable = any(x in error_msg for x in ["503", "429", "500", "NoneType", "empty", "timeout", "deadline"])
                if attempt == self.max_retries or not is_retryable:
                    raise
                wait = 2 ** attempt
                print(f"      Retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError("Unreachable")

    # ------------------------------------------------------------
    # 4.  Public entry point – automatically selects method
    # ------------------------------------------------------------
    def summarize(self, transcript_text: str, metadata: dict) -> str:
        word_count = len(transcript_text.split())
        if word_count > self.long_video_threshold:
            return self._summarize_long(transcript_text, metadata)
        else:
            return self._summarize_direct(transcript_text, metadata)

# ===================== FILE EXISTENCE CHECK =====================
def should_skip_video(video_id: str) -> bool:
    """Return True if all output files for this video already exist."""
    raw_path = os.path.join(RAW_TRANSCRIPTS_DIR, f"{video_id}.json")
    clean_path = os.path.join(CLEAN_DIR, f"{video_id}_cleaned.txt")
    summary_path = os.path.join(GEMINI_SUMMARY_DIR, f"{video_id}_summary.txt")
    return all(os.path.exists(p) for p in [raw_path, clean_path, summary_path])

# ===================== MAIN PIPELINE =====================
def process_video(video_id: str, 
                  extractor: YouTubeTranscriptExtractor,
                  cleaner: TranscriptCleaner,
                  summarizer: GeminiSummarizer,
                  force: bool = False) -> bool:
    """
    Process a single video from extraction to summary.
    Skips if all output files already exist (unless force=True).
    """
    if not force and should_skip_video(video_id):
        print(f"\n Skipping {video_id} – all output files already exist.")
        return True

    print(f"\n{'='*60}")
    print(f"Processing: {video_id}")
    print(f"{'='*60}")
    
    # ---- 1. Extract ----
    raw_path = os.path.join(RAW_TRANSCRIPTS_DIR, f"{video_id}.json")
    if os.path.exists(raw_path) and not force:
        print("1. Loading existing raw transcript...")
        with open(raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print("1. Extracting transcript...")
        data = extractor.extract(video_id)
        if not data:
            return False
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved raw JSON")

    print(f"   Title: {data['title'][:70]}...")
    print(f"   Words: {data.get('word_count', 0):,}")

    # ---- 2. Clean ----
    clean_path = os.path.join(CLEAN_DIR, f"{video_id}.txt")
    if os.path.exists(clean_path) and not force:
        print("\n2. Loading existing cleaned text...")
        with open(clean_path, "r", encoding="utf-8") as f:
            cleaned_text = f.read()
    else:
        print("\n2. Cleaning transcript...")
        transcript_segments = [seg["text"] for seg in data.get("transcript", [])]
        full_text = " ".join(transcript_segments)
        cleaned_text = cleaner.clean_text(full_text)
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"   Saved cleaned text")

    # ---- 3. Summarize ----
    summary_path = os.path.join(GEMINI_SUMMARY_DIR, f"{video_id}_summary.txt")

    # If force=False and summary already exists, skip summarisation entirely
    if not force and os.path.exists(summary_path):
        print("\n3. Summary already exists – skipping summarization.")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = f.read()
    else:
        print("\n3. Summarizing with Gemini...")
        try:
            summary = summarizer.summarize(cleaned_text, data)
        except Exception as e:
            print(f"\n Summarization failed for {video_id}: {e}")
            return False  # not write a summary file
        
        # Write summary only if summarization succeeded
        metadata_header = f"""VIDEO: {data.get('title', 'Unknown')}
    CHANNEL: {data.get('channel', 'Unknown')}
    PUBLISHED: {data.get('published', 'Unknown')[:10]}
    VIDEO ID: {video_id}
    TRANSCRIPT WORDS: {data.get('word_count', 0):,}
    {'='*60}

    """
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(metadata_header + summary)
        print(f"   Summary saved")


def main():
    init_directories()

    # API key checks
    if not YOUTUBE_API_KEY:
        print("Error: YOUTUBE_API_KEY not set in .env")
        return
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set in .env")
        print("Get one from: https://aistudio.google.com/app/apikey")
        return

    # Instantiate components
    extractor = YouTubeTranscriptExtractor(YOUTUBE_API_KEY)
    cleaner = TranscriptCleaner()
    summarizer = GeminiSummarizer(GEMINI_API_KEY, model_name="gemini-2.5-pro")

    # --- List video IDs here ---
    video_ids = [
        # Paste actual IDs
        'dXXRD3zxa6Y',
        'Ul35-p7riEo',
        'kKYEQyJ1GLc',
        '0RZczdJeTZ0'
    ]

    if not video_ids:
        print("Please add video IDs to the `video_ids` list in main().")
        return


    # Process each video – skip if already done
    results = []
    for vid in video_ids:
        success = process_video(vid, extractor, cleaner, summarizer, force=False)
        results.append((vid, success))
        if vid != video_ids[-1]:
            time.sleep(1)  # delay

    # Final report
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    successful = sum(1 for _, s in results if s)
    print(f"\n Successfully processed: {successful}/{len(video_ids)} videos")
    print(f"\n Output folders:")
    print(f"   {RAW_TRANSCRIPTS_DIR}/  (raw JSON)")
    print(f"   {CLEAN_DIR}/            (.txt)")
    print(f"   {GEMINI_SUMMARY_DIR}/   (final summaries)")

if __name__ == "__main__":
    main()