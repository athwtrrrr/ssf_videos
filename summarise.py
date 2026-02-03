from transformers import pipeline
import os

SUMMARY_DIR = "data/summaries"
os.makedirs(SUMMARY_DIR, exist_ok=True)

# Summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text: str, max_len=150, min_len=40) -> str:
    return summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']

CHUNK_DIR = "data/chunks"

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

print("Summarization complete.")
