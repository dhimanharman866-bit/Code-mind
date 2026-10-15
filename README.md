<img width="306" height="1897" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/7caeee62-6597-4e87-a23b-93085dfeefc3" />
# CodeMind — RAG-Powered Codebase Assistant

An intelligent codebase Q&A system that uses a multi-stage Retrieval-Augmented Generation (RAG) pipeline to let you ask natural language questions about any codebase and get precise, code-grounded answers.

---

## Overview

Modern codebases are large, fragmented, and hard to reason about quickly. CodeMind solves this by indexing your entire project into a vector database and combining multi-query retrieval, cross-encoder reranking, and an LLM to return accurate, cited answers — not hallucinations.

Upload a ZIP or paste a GitHub URL, and start asking questions immediately.

---

## RAG Pipeline

The pipeline has 6 stages, each handled by a dedicated module:

```
Codebase (ZIP / GitHub)
        │
        ▼
  1. loader.py        →  Walks the folder, reads all supported files,
                         skips noise (node_modules, __pycache__, .git),
                         tracks file hashes to detect changes
        │
        ▼
  2. chunker.py       →  Language-aware splitting (Python, JS, Java, C++…)
                         using RecursiveCharacterTextSplitter
                         Chunk size: 1500 chars | Overlap: 200 chars
        │
        ▼
  3. embeddings.py    →  Converts chunks to dense vectors using
                         BAAI/bge-large-en-v1.5 (1024-dim, normalized)
        │
        ▼
  4. vectorstore.py   →  Stores vectors in a FAISS flat index
                         Saved to disk and reloaded on restart
        │
        ▼
  5. retriever.py     →  MultiQueryRetriever: generates 3 rephrasings
                         of the user query using llama-3.1-8b-instant,
                         then runs MMR search (k=8, fetch_k=25) for each
        │
        ▼
  6. reranker.py      →  Cross-encoder (ms-marco-MiniLM-L-6-v2) re-scores
                         all retrieved chunks by reading query + chunk
                         together → keeps top 5 highest-quality chunks
        │
        ▼
  7. qa_chain.py      →  Top 5 chunks sent to Groq LLM with structured
                         prompt → cited, file-grounded answer returned
```

---

## Key Features

- **Multi-query retrieval** — automatically generates multiple search queries from one user question, improving recall for natural language questions that don't literally match code tokens
- **Cross-encoder reranking** — re-scores retrieved chunks using a dedicated reranker model before sending to the LLM, cutting noise and improving precision
- **Language-aware chunking** — uses language-specific separators for Python, JavaScript, Java, C++, and more to avoid splitting mid-function
- **Large embedding model** — `BAAI/bge-large-en-v1.5` (1024-dim) vs. a small model, significantly better at matching semantic intent to code
- **Source citations** — every answer shows which files were referenced, with expandable code previews
- **Chat history** — multi-turn conversation with memory of previous questions
- **ZIP + GitHub support** — upload a ZIP or paste any public GitHub repo URL to index
- **File hash tracking** — detects changed files between sessions

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Query generation | Groq (`openai/gpt-oss-20b`) |
| Embedding model | `BAAI/bge-large-en-v1.5` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector store | FAISS (CPU) |
| RAG framework | LangChain |
| Chunking | `RecursiveCharacterTextSplitter` |

---

## Project Structure

```
.
├── app.py              # Streamlit UI — upload, chat, source viewer
├── loader.py           # File ingestion, hash tracking, noise filtering
├── chunker.py          # Language-aware code splitting
├── embeddings.py       # HuggingFace embedding model setup
├── vectorstore.py      # FAISS index build & persist
├── retriever.py        # MultiQueryRetriever with MMR base
├── reranker.py         # Cross-encoder reranking (top-5 selection)
├── qa_chain.py         # LLM chain with reranking + prompt
├── config.py           # All tunable settings in one place
├── main.py             # CLI entry point
├── requirements.txt
└── .gitignore
```

---

## Configuration

All key settings live in [`config.py`](config.py):

```python
CHUNK_SIZE   = 1500                      # chars per chunk
CHUNK_OVERLAP = 200                      # overlap between chunks
EMBED_MODEL  = "BAAI/bge-large-en-v1.5" # embedding model
INDEX_PATH   = "faiss_index_new"         # FAISS index directory
HASH_FILE    = "file_hashes.json"        # change-detection cache
```

---

## Supported File Types

`.py` `.js` `.ts` `.jsx` `.tsx` `.java` `.cpp` `.c` `.h` `.hpp`
`.go` `.rs` `.rb` `.php` `.cs` `.html` `.css` `.md` `.txt`
`.yaml` `.yml` `.json` `.toml`

---

## Installation

```bash
git clone https://github.com/dhimanharman866-bit/rag-code-analyzer.git
cd rag-code-analyzer
pip install -r requirements.txt
```

Set your Groq API key:

```bash
# Option A — .env file
echo "GROQ_API_KEY=your_key_here" > .env

# Option B — Streamlit secrets
# Add GROQ_API_KEY to .streamlit/secrets.toml
```

---

## Usage

```bash
streamlit run app.py
```

1. Open the app in your browser
2. Upload a **ZIP file** or enter a **GitHub repo URL** in the sidebar
3. Click **Index Codebase** — models download automatically on first run
4. Ask any question about the code in the chat

> **Note:** On first run, two models are downloaded automatically:
> - `BAAI/bge-large-en-v1.5` (~1.3 GB) — embedding model
> - `cross-encoder/ms-marco-MiniLM-L-6-v2` (~85 MB) — reranker

---

## Example Questions

- *"How does authentication work in this project?"*
- *"Where is the database connection initialized?"*
- *"Explain the flow from user input to API response"*
- *"What does the `UserService` class do?"*
- *"Why might this module throw a NullPointerException?"*
- *"List all the API endpoints defined in this codebase"*

---

## Important Notes

- After changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBED_MODEL`, delete the `faiss_index_new/` folder and `file_hashes.json` before re-running — stale indexes will silently produce wrong results
- Large repositories (1000+ files) may take a few minutes to index on first run
- Requires a valid Groq API key for LLM responses

---

## Author

Harmanpreet Dhiman  
AI + Software Engineering Enthusiast

---

## License

This project is open-source and available under the MIT License.
