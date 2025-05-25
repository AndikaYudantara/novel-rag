import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

# CONFIG
chunked_file = "data/chunked_novel.jsonl"
faiss_index_file = "embeddings/faiss_index.bin"
embedding_model_name = "all-MiniLM-L6-v2"
embedding_dim = 384  # embedding size of the model

def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    print("Loading embedding model on GPU...")
    model = SentenceTransformer(embedding_model_name, device="cuda") # Change to "cuda" if you have a GPU

    print("Loading chunks...")
    chunks = list(load_chunks(chunked_file))
    print(f"Loaded {len(chunks)} chunks.")

    # Create FAISS index (Flat L2)
    index = faiss.IndexFlatL2(embedding_dim)

    # Map from FAISS idx to chunk metadata
    id_to_metadata = {}

    embeddings = []

    print("Embedding chunks...")
    for i, chunk in enumerate(tqdm(chunks)):
        emb = model.encode(chunk["text"], convert_to_numpy=True)
        embeddings.append(emb)
        id_to_metadata[i] = {
            "chunk_id": chunk["chunk_id"],
            "chapter": chunk["chapter"],
            "text": chunk["text"],
        }

    embeddings = np.vstack(embeddings).astype("float32")

    print("Adding embeddings to FAISS index...")
    index.add(embeddings)

    # Save index
    Path("embeddings").mkdir(exist_ok=True)
    faiss.write_index(index, faiss_index_file)

    # Save metadata
    with open("embeddings/id_to_metadata.json", "w", encoding="utf-8") as f:
        json.dump(id_to_metadata, f, ensure_ascii=False, indent=2)

    print(f"Done! FAISS index and metadata saved to 'embeddings/'.")

if __name__ == "__main__":
    main()
