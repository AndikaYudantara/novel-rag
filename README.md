# Novel-RAG

A Retrieval-Augmented Generation (RAG) pipeline for question answering over long-form novel text using chunking, embedding, FAISS vector search, and LLMs (e.g., OpenAI's GPT-4 or HuggingFace's Flan-T5).

## Features

- **Text Chunking:** Splits large novel files into overlapping, context-preserving chunks.
- **Embedding:** Uses Sentence Transformers to embed text chunks.
- **FAISS Indexing:** Stores embeddings for fast similarity search.
- **Question Answering:** Retrieves relevant chunks and generates answers using an LLM (OpenAI or HuggingFace).
- **Modular:** Easily switch between OpenAI and open-source LLMs.

## Project Structure

The project is organized as follows:

- `data/`: Directory to store your novel `.txt` files.
- `app/chunker.py`: Script to split large text files into chunks.
- `app/embed_faiss.py`: Script to generate embeddings and create a FAISS index.
- `app/qa_openai.py`: Script to perform question answering using OpenAI's API.
- `requirements.txt`: File listing the Python dependencies for the project.

## Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Prepare data:**

    Place your novel .txt files in the data/ directory.

3. **Chunk data:**

    Run the chunker to process your novel files:
    ```sh
    python app/chunker.py
    ```

4. **Embed and index data:**

    Generate embeddings for the chunks and create a FAISS index:
    ```sh
    python app/embed_faiss.py
    ```

## Configuration
    - Adjust chunk size and overlap in `chunker.py`.
    - Change embedding model in `embed_faiss.py` and QA scripts.

    
## Notes
        Requires a CUDA-capable GPU for embedding and LLM inference.
        For users without GPU access, the scripts can fall back to CPU execution, though it may be slower.
        Alternatively, consider using cloud-based services like Google Colab or AWS for GPU access.
    
    
## License

    This project is licensed under the MIT License. You are free to use, modify, and distribute the code, provided proper attribution is given. The software is provided "as-is" without any warranties.
    
    For more details, see the full license text [here](https://opensource.org/licenses/MIT).
    License
        MIT License
