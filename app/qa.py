import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load FAISS
index = faiss.read_index("faiss_index/index.faiss")

# Load metadata (original text chunks)
with open("faiss_index/chunk_metadata.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_model.to("cuda")

# Load LLM (Open source or HuggingFace)
qa_model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device=0)

# Ask a question
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    # Embed the query
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding).astype("float32"), k=5)

    # Retrieve top chunks
    context = "\n\n".join(chunks[i] for i in I[0])

    # Prompt
    prompt = f"Answer this question based on the following text:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    # Get answer from LLM
    answer = qa_model(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text']
    print("\n" + answer.strip())
