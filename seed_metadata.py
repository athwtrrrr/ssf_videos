import json
import os

def seed_metadata_minimal():
    
    RAW_TRANSCRIPTS_DIR = "data/transcripts"
    
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
            for file in os.listdir(dir_name):
                if file.startswith(video_id) and file.endswith(".txt"):
                    file_path = os.path.join(dir_name, file)
                    
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(tag + content)
        
        print(f" Tagged files for: {metadata.get('title', 'Unknown')[:30]}...")

if __name__ == "__main__":
    seed_metadata_minimal()