import os
import re
import json
import time
import random
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline
from dotenv import load_dotenv

# ===================== CONFIGURATION =====================
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Directories 
RAW_TRANSCRIPTS_DIR = "data/transcripts"
CLEAN_DIR = "data/cleaned"
CHUNK_DIR = "data/chunks"
SUMMARY_DIR = "data/summaries"
FINAL_SUMMARY_DIR = "data/final"
COMBINED_SUMMARY_DIR = "data/combined_summarised_chunks"

def init_directories():
    """Create all necessary directories"""
    for dir_path in [RAW_TRANSCRIPTS_DIR, CLEAN_DIR, CHUNK_DIR, 
                     SUMMARY_DIR, FINAL_SUMMARY_DIR, COMBINED_SUMMARY_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    print("✓ All directories created")

# ===================== EXTRACTION MODULE =====================
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

# ===================== CLEANING MODULE =====================
class TranscriptCleaner:
    """
    Exact copy from clean.py
    """
    def __init__(self):
        # Common fillers, background noises, and brackets
        self.filler_patterns = [
            r"\bum+\b",
            r"\buh+\b",
            r"\[music\]",
            r"\[applause\]",
            r"\[laughter\]",
            r"\[.*?\]"  
        ]
    
    def clean_text(self, text: str) -> str:
        # 1. Remove filler words
        for pattern in self.filler_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # 2. Remove duplicated words
        text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)

        # 3. Fix spacing issues
        text = re.sub(r"\s+", " ", text).strip()

        # 4. Capitalize sentences lightly
        text = self.capitalize_sentences(text)

        return text

    def capitalize_sentences(self, text: str) -> str:
        # Split by sentence-ending punctuation
        sentences = re.split(r'([.!?])', text)
        cleaned_sentences = []
        for i in range(0, len(sentences)-1, 2):
            sentence = sentences[i].strip()
            punctuation = sentences[i+1]
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
                cleaned_sentences.append(sentence + punctuation)
        return ' '.join(cleaned_sentences)

    def clean_transcript_json(self, transcript: List[Dict]) -> str:
        texts = [segment.get("text", "") for segment in transcript]
        full_text = " ".join(texts)
        return self.clean_text(full_text)

    def chunk_transcript(self, text: str, max_words: int = 1500, overlap: int = 100) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + max_words
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            start = end - overlap  # overlap to preserve context
        return chunks

# ===================== SUMMARIZATION MODULES =====================
def summarize_text(text: str, max_len=150, min_len=40) -> str:
    """
    Exact function from sum_raw_chunks.py
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']

def extract_video_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract video ID 
    """
    patterns = [
        r'^(.*?)_chunk\d+_summary\.txt$',  # videoId_chunkX_summary.txt
    ]
    
    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            return match.group(1)
    
    # If no pattern matches, remove _summary.txt and see if it's a direct video ID
    if filename.endswith("_summary.txt"):
        return filename.replace("_summary.txt", "")
    
    return None

def get_video_metadata(video_id: str) -> Dict:
    """
    Retrieve video metadata from raw transcripts 
    """
    metadata_path = os.path.join(RAW_TRANSCRIPTS_DIR, f"{video_id}.json")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "title": data.get("title", "Unknown Title"),
                    "channel": data.get("channel", "Unknown Channel"),
                    "published": data.get("published", "Unknown Date"),
                    "transcript_chunks": len(data.get("transcript", []))
                }
        except Exception as e:
            print(f"Error reading metadata for {video_id}: {e}")
    
    return {
        "title": f"Video {video_id}",
        "channel": "Unknown Channel",
        "published": "Unknown Date",
        "transcript_chunks": 0
    }

def combine_summaries_with_intelligence(summaries: List[str]) -> str:
    """
    Combine summaries (exact from sum_combined_chunks.py)
    """
    if not summaries:
        return ""
    
    if len(summaries) == 1:
        return summaries[0]
    
   
    combined = []
    
    for summary in summaries:
        # Clean up each summary
        summary = summary.strip()
        
        # Ensure proper punctuation
        if summary and not summary.endswith(('.', '!', '?')):
            summary = summary + '.'
        
        # Capitalize if needed
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
        
        combined.append(summary)
    
    # Join with double newlines for clear separation
    return "\n\n".join(combined)

class HierarchicalSummarizer:
    """
    Stage 2 summarizer for generating final video summaries
    """
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        print(f"Loading model: {model_name}...")
        self.summarizer = pipeline("summarization", model=model_name)
        print("Model loaded successfully!")
    
    def generate_final_summary(self, combined_text: str, video_length: int = None) -> str:
        """
        Generate final summary based on combined chunk summaries
        """
        if not combined_text or len(combined_text.strip()) < 50:
            return "Insufficient content for summarization."
        
        # Calculate optimal summary length based on input
        word_count = len(combined_text.split())
        
        if word_count > 2000:
            max_len = 200
            min_len = 100
        elif word_count > 1000:
            max_len = 150
            min_len = 75
        elif word_count > 500:
            max_len = 120
            min_len = 60
        elif word_count > 200:
            max_len = 80
            min_len = 40
        else:
            max_len = 60
            min_len = 30
        
        # Adjust based on video length if provided
        if video_length and video_length > 30:  # Long video
            max_len = min(max_len + 50, 250)
            min_len = min(min_len + 25, 125)
        
        try:
            print(f"Generating summary (input: {word_count} words, target: {min_len}-{max_len} words)...")
            
            summary = self.summarizer(
                combined_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
                no_repeat_ngram_size=3  # Avoid repetition
            )[0]['summary_text']
            
            # Clean up the summary
            summary = self.post_process_summary(summary)
            
            return summary
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            # Fallback: return the first few sentences
            sentences = combined_text.split('. ')
            return ". ".join(sentences[:3]) + "."
    
    def post_process_summary(self, summary: str) -> str:
        """
        Clean up and improve the summary text
        """
        # Remove redundant phrases
        redundant = [
            "This video discusses ",
            "The speaker talks about ",
            "In this video, ",
            "This summary covers ",
            "The content focuses on "
        ]
        
        for phrase in redundant:
            if summary.startswith(phrase):
                summary = summary[len(phrase):]
                # Capitalize first letter
                if summary:
                    summary = summary[0].upper() + summary[1:]
        
        # Ensure it ends with proper punctuation
        if not summary.endswith(('.', '!', '?')):
            summary = summary.rstrip() + '.'
        
        return summary

def seed_metadata_minimal():
    """
    Exact function from seed_metadata.py
    """
    for json_file in os.listdir(RAW_TRANSCRIPTS_DIR):
        if not json_file.endswith(".json"):
            continue
        
        video_id = json_file.replace(".json", "")
        
        # Load metadata
        with open(os.path.join(RAW_TRANSCRIPTS_DIR, json_file), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Create tag line
        tag = f"# VIDEO: {metadata.get('title', '')} | CHANNEL: {metadata.get('channel', '')} | DATE: {metadata.get('published', '')[:10]}\n\n"
        
        # Add to chunk files
        for dir_name in ["data/chunks", "data/summaries", "data/cleaned"]:
            dir_path = dir_name
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.startswith(video_id) and file.endswith(".txt"):
                        file_path = os.path.join(dir_path, file)
                        
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(tag + content)
        
        print(f" Tagged files for: {metadata.get('title', 'Unknown')[:30]}...")

# ===================== MAIN PIPELINE FUNCTIONS =====================
def run_extraction(video_ids: List[str]):

    if not YOUTUBE_API_KEY:
        print("Error: YouTube API key not found!")
        print("Please set YOUTUBE_API_KEY in environment variables or .env file")
        return False
    
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
    
    return True

def run_cleaning_and_chunking():

    cleaner = TranscriptCleaner()

    for file_name in os.listdir(RAW_TRANSCRIPTS_DIR):
        if not file_name.endswith(".json"):
            continue
        
        file_path = os.path.join(RAW_TRANSCRIPTS_DIR, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Join transcript segments
        transcript_text = " ".join([seg["text"] for seg in data.get("transcript", [])])
        
        # Clean
        cleaned_text = cleaner.clean_text(transcript_text)

        # Save cleaned transcript
        clean_file_path = os.path.join(CLEAN_DIR, file_name.replace(".json", ".txt"))
        with open(clean_file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        # Chunk
        chunks = cleaner.chunk_transcript(cleaned_text, max_words=1500, overlap=100)
        
        # Save chunks
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(CHUNK_DIR, f"{file_name.replace('.json','')}_chunk{i+1}.txt")
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk)

    print("Cleaning and chunking complete. Chunks are ready for summarization.")
    return True

def run_chunk_summarization():

    os.makedirs(SUMMARY_DIR, exist_ok=True)
    
    # Summarize all chunks
    for chunk_file in os.listdir(CHUNK_DIR):
        if not chunk_file.endswith(".txt"):
            continue
        path = os.path.join(CHUNK_DIR, chunk_file)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        summary = summarize_text(text)
        
        summary_path = os.path.join(SUMMARY_DIR, chunk_file.replace(".txt", "_summary.txt"))
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

    print("Chunk summarization complete.")
    return True

def run_final_summarization():

    os.makedirs(FINAL_SUMMARY_DIR, exist_ok=True)
    os.makedirs(COMBINED_SUMMARY_DIR, exist_ok=True)
    
    def group_summaries_by_video() -> Dict[str, Dict]:
        """
        Group all chunk summaries by video ID and collect metadata
        """
        video_data = {}
        
        for filename in os.listdir(SUMMARY_DIR):
            if not filename.endswith(".txt"):
                continue
            
            video_id = extract_video_id_from_filename(filename)
            if not video_id:
                print(f"Warning: Could not extract video ID from {filename}")
                continue
            
            if video_id not in video_data:
                video_data[video_id] = {
                    "summaries": [],
                    "chunk_count": 0,
                    "files": []
                }
            
            # Read the summary content
            file_path = os.path.join(SUMMARY_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    
                if content:
                    video_data[video_id]["summaries"].append(content)
                    video_data[video_id]["chunk_count"] += 1
                    video_data[video_id]["files"].append(filename)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        
        return video_data
    
    def save_final_results(video_id: str, metadata: Dict, final_summary: str, 
                          combined_text: str, chunk_count: int):
        """
        Save all final results and intermediate data
        """
        # 1. Save final summary
        final_path = os.path.join(FINAL_SUMMARY_DIR, f"{video_id}_final_summary.txt")
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(f"Video: {metadata['title']}\n")
            f.write(f"Channel: {metadata['channel']}\n")
            f.write(f"Published: {metadata['published']}\n")
            f.write(f"Original Transcript Chunks: {metadata['transcript_chunks']}\n")
            f.write(f"Summary Chunks Combined: {chunk_count}\n")
            f.write("=" * 60 + "\n\n")
            f.write(final_summary)
        
        # 2. Save combined summaries (for reference)
        combined_path = os.path.join(COMBINED_SUMMARY_DIR, f"{video_id}_combined_summaries.txt")
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write(f"Combined chunk summaries for: {video_id}\n")
            f.write("=" * 60 + "\n\n")
            f.write(combined_text)
        
        return final_path
    
    # Step 1: Group all existing chunk summaries by video
    print("\nStep 1: Grouping chunk summaries by video...")
    video_data = group_summaries_by_video()
    
    if not video_data:
        print("No chunk summaries found! Please run Stage 1 first.")
        return False
    
    print(f"Found summaries for {len(video_data)} videos")
    for vid, data in video_data.items():
        print(f"  - {vid}: {data['chunk_count']} chunk summaries")
    
    # Step 2: Initialize the summarizer
    print("\nStep 2: Initializing summarizer...")
    summarizer = HierarchicalSummarizer()
    
    # Step 3: Process each video
    print("\nStep 3: Generating final summaries...")
    print("-" * 60)
    
    results = []
    
    for video_id, data in video_data.items():
        print(f"\nProcessing: {video_id}")
        print(f"  Chunks to combine: {data['chunk_count']}")
        
        # Get video metadata
        metadata = get_video_metadata(video_id)
        print(f"  Title: {metadata['title'][:50]}...")
        
        # Combine summaries 
        combined_text = combine_summaries_with_intelligence(data['summaries'])
        print(f"  Combined text: {len(combined_text.split())} words")
        
        # Generate final summary
        final_summary = summarizer.generate_final_summary(
            combined_text, 
            video_length=metadata['transcript_chunks']
        )
        
        # Save results
        output_path = save_final_results(
            video_id, 
            metadata, 
            final_summary, 
            combined_text, 
            data['chunk_count']
        )
        
        results.append({
            "video_id": video_id,
            "title": metadata['title'],
            "output_path": output_path,
            "summary_preview": final_summary[:100] + "..." if len(final_summary) > 100 else final_summary
        })
        
        print(f"  ✓ Final summary saved")
    
    return True

# ===================== MAIN EXECUTION =====================
def main():
    """
    Main function to run the complete pipeline
    """
    # Initialize directories
    init_directories()
    
    # Get video IDs 
    video_ids = [
        "xD_I489cf64"
    ]
    
    if not video_ids:
        print("Please add video IDs to the video_ids list in the main() function.")
        return
    
    # Step 1: Extract transcripts
    print("\nSTEP 1: Extracting transcripts...")
    run_extraction(video_ids)
    
    # Step 2: Clean and chunk
    print("\nSTEP 2: Cleaning and chunking...")
    run_cleaning_and_chunking()
    
    # Step 3: Summarize chunks
    print("\nSTEP 3: Summarizing chunks...")
    run_chunk_summarization()
    
    # Step 4: Create final summaries
    print("\nSTEP 4: Creating final summaries...")
    run_final_summarization()
    
    # Step 5: Tag files with metadata
    print("\nSTEP 5: Tagging files with metadata...")
    seed_metadata_minimal()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nOutput directories:")
    print(f"  Raw transcripts: {RAW_TRANSCRIPTS_DIR}")
    print(f"  Cleaned text: {CLEAN_DIR}")
    print(f"  Chunks: {CHUNK_DIR}")
    print(f"  Chunk summaries: {SUMMARY_DIR}")
    print(f"  Combined summaries: {COMBINED_SUMMARY_DIR}")
    print(f"  Final summaries: {FINAL_SUMMARY_DIR}")

if __name__ == "__main__":
    
    main()