
# 📚 NovelRAG: QA System for Long Novels using Retrieval-Augmented Generation

A semantic search and question-answering tool built for novels with 2000+ chapters. This project uses **FAISS** for vector search and **OpenAI GPT** (or other LLMs) to answer questions using relevant content from your novel.

---

## 🛠 Features

  - 📖 Process and index thousands of novel chapters
  - 🔍 Search relevant content using semantic similarity
  - 💬 Ask natural language questions about the story
  - 🧠 Uses OpenAI for intelligent, context-based answers
  - ⚡ Fast local search using FAISS (CPU or GPU)

---
## 🔧 Tested Hardware
  - CPU: AMD Ryzen 5 7500F
  - GPU: RTX 3070 Ti

⚠️ **You may using different version of cuda if you have other GPU**

---

## 📁 Project Structure

```
novel-rag/
├── app/
│   ├── chunker.py          # Splits novel into clean chapters
│   ├── embed.py            # Embeds chunks using SentenceTransformer
│   ├── build_faiss.py      # Creates and saves FAISS index
│   ├── qa_openai.py        # Question-answering pipeline using OpenAI
│   └── utils.py            # Helper functions
├── data/                   # Raw novel .txt files
├── embeddings/             # Saved FAISS index and metadata
├── .env                    # API key and config (not tracked)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/novel-rag.git
cd novel-rag
```

### 2. Install Python 3.10–3.11

⚠️ **The Latest Release: Python 3.13 is not compatible with many packages (like `torch` and `faiss`)**

> Use [pyenv](https://github.com/pyenv/pyenv) or install manually from https://www.python.org/

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If using GPU:

- Install PyTorch with CUDA from https://pytorch.org/get-started/locally/
- GPU FAISS (if compatible): `pip install faiss-gpu`
- Or CPU FAISS: `pip install faiss-cpu`
  
⚠️ **If you using windows, it is recommended to use cpu for FAISS, because the GPU setup is more complicated**

---

## 🔑 Environment Variables

Create a `.env` file in the root folder:

```env
OPENAI_API_KEY=your_openai_key_here
```

---

## 🧩 Usage Guide

### Step 1: Preprocess Chapters

Put your `.txt` novel files in the `data/` directory.

Run:

```bash
python app/chunker.py
```

This splits your novel into chapters, cleans text, and prepares chunks.

---

### Step 2: Generate Embeddings

```bash
python app/embed.py
```

This creates semantic embeddings for each chunk and saves them to disk.

---

### Step 3: Build FAISS Index

```bash
python app/build_faiss.py
```

This builds the FAISS vector index to enable fast search.

---

### Step 4: Ask Questions

```bash
python app/qa_openai.py
```

Example prompt:

```
> What is the Northern End Cave and what did Grid find there?
```

You’ll get a response based on semantically relevant chunks of the novel.

---

## 💡 Notes

- If you get a `KeyError` or `IndexError`, your FAISS index may not align with chunk metadata.
- Output may not always show chapter titles — you can modify the QA script to include this metadata.

---

## 🔮 Future Ideas

- [ ] Web UI using Streamlit or React
- [ ] Display exact chapter/source info for answers
- [ ] Replace OpenAI with local models (e.g. Mistral, Phi-2, LLaMA)
- [ ] Multilingual input/novel support


## 📜 License

MIT License — you are free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- Created by **Yudantara**  
- Powered by OpenAI & FAISS.
- Special Thanks to the [Overgeared](https://www.wuxiaworld.com/novel/overgeared) novel universe with Park Saenal (박새날) as author and an amazing translation by rainbowturtle ❤️

