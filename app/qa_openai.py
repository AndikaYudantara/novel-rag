import faiss
import numpy as np
import json
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


# 🗝️ Set your API key (or load from environment variable)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# 📦 Load FAISS index and chunks
index = faiss.read_index("embeddings/faiss_index.bin")

with open("embeddings/id_to_metadata.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# 📐 Load sentence transformer for embedding
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_model.to("cuda")

with open("embeddings/id_to_metadata.json", "r", encoding="utf-8") as f:
    chunks_dict = json.load(f)
# Convert chunks_dict to a list for easier access
chunks = [chunks_dict[str(i)] for i in range(len(chunks_dict))]

def get_relevant_chunks(question, k=20):
    q_emb = embed_model.encode([question]).astype("float32")
    D, I = index.search(q_emb, k)
    valid_chunks = []
    for i in I[0]:
        valid_chunks.append(chunks[i])
    return valid_chunks


def ask_openai(question, context, model="gpt-4o"):
    prompt = f"""You are an assistant helping answer questions about a novel.
Answer the following question using the provided context.

Context:
{context}

Question: {question}
Answer:"""

    
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=0.5,   
    )

    return response.output_text 

# 🧪 Main loop
if __name__ == "__main__":
    print("Ask a question about your novel! Type 'exit' to quit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == "exit":
            break

        relevant = get_relevant_chunks(question)
        context = "\n\n".join(
            f"[{chunk['chapter']}\n{chunk['text']}]" 
            for chunk in relevant 
        )
        if not context:
            print("No relevant context found. Please try a different question.")
            continue

        print(context)
        print("\nThinking...\n")
        answer = ask_openai(question, context)
        print("📘 Answer:\n" + answer.strip())
