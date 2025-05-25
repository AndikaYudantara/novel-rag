import re
import json
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
input_files = [
    "data/OVERGEARED_PART_1.txt",
    "data/OVERGEARED_PART_2.txt",
    "data/OVERGEARED_PART_3.txt",
]
output_file = "data/chunked_novel.jsonl"
chunk_max_words = 500  # You can adjust to 300 or 200 if needed
chunk_overlap = 100     # Helps preserve context across chunks

# === FUNCTIONS ===

def split_by_chapters(text):
    """Split text into (title, body) pairs using 'Chapter X' pattern."""
    parts = re.split(r"(Chapter\s+\d+[:]?.*)", text)
    result = []
    title_then = None
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        if title_then == title:
            continue
        body = parts[i+1].strip()
        title_then = title
        result.append((title, body))
    return result

def chunk_text(text, max_words=250, overlap=50):
    """Chunk paragraph text into overlapping word chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = words[i:i + max_words]
        if len(chunk) < 10:
            continue  # Skip too-short chunks
        chunks.append(" ".join(chunk))
    return chunks

# === MAIN ===

def main():
    chunk_id = 0
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_file:
        for file_index, file_path in enumerate(input_files):
            print(f"🔍 Processing {file_path}...")
            full_text = Path(file_path).read_text(encoding="utf-8")
            chapters = split_by_chapters(full_text)

            for chapter_num, (title, body) in tqdm(enumerate(chapters, start=1), desc="Chapters"):
                chunks = chunk_text(body, chunk_max_words, chunk_overlap)
                for chunk in chunks:
                    chunk_id += 1
                    record = {
                        "chunk_id": f"novel{file_index+1}_{chunk_id:05}",
                        "chapter": title,
                        "text": chunk
                    }
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Done! Saved {chunk_id} chunks to {output_file}")

if __name__ == "__main__":
    main()
