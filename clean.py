import os
import re
import json
from typing import List, Dict

class TranscriptCleaner:
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

    def chunk_transcript(self,text: str, max_words: int = 1500, overlap: int = 100) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + max_words
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            start = end - overlap  # overlap to preserve context
        return chunks
# -------------------------
# Pipeline
# -------------------------
RAW_DIR = "data/transcripts"
CLEAN_DIR = "data/cleaned"
CHUNK_DIR = "data/chunks"

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

cleaner = TranscriptCleaner()

for file_name in os.listdir(RAW_DIR):
    if not file_name.endswith(".json"):
        continue
    
    file_path = os.path.join(RAW_DIR, file_name)
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