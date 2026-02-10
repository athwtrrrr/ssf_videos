import os
import json
import re
from typing import Dict, List, Optional
from transformers import pipeline

# Directories
CHUNK_SUMMARY_DIR = "data/summaries"
RAW_TRANSCRIPTS_DIR = "data/transcripts"  # For metadata
FINAL_SUMMARY_DIR = "data/final"
COMBINED_SUMMARY_DIR = "data/combined_summarised_chunks"
os.makedirs(FINAL_SUMMARY_DIR, exist_ok=True)
os.makedirs(COMBINED_SUMMARY_DIR, exist_ok=True)

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

def group_summaries_by_video() -> Dict[str, Dict]:
    """
    Group all chunk summaries by video ID and collect metadata
    Returns: {video_id: {"summaries": [], "chunk_count": 0, "files": []}}
    """
    video_data = {}
    
    for filename in os.listdir(CHUNK_SUMMARY_DIR):
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
        file_path = os.path.join(CHUNK_SUMMARY_DIR, filename)
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
    Combine summaries
    """
    if not summaries:
        return ""
    
    if len(summaries) == 1:
        return summaries[0]
    
    # Simply join all summaries with paragraph breaks
    # No need for complex logic since there's no overlap
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

def main():
    """
    Main execution for Stage 2 hierarchical summarization
    """
    # Step 1: Group all existing chunk summaries by video
    print("\nStep 1: Grouping chunk summaries by video...")
    video_data = group_summaries_by_video()
    
    if not video_data:
        print("No chunk summaries found! Please run Stage 1 first.")
        return
    
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

if __name__ == "__main__":
    main()