# 🧠 RAG Document Intelligence System

Upload PDFs and ask questions — get accurate, source-cited answers grounded entirely in your documents.


---

## What this is

I built this because I was tired of two things: Ctrl+F not being smart enough, and ChatGPT confidently making things up when I asked about my own documents.

This app lets you upload any PDF — earnings reports, research papers, policy docs, whatever — and ask it questions in plain English. It pulls the exact relevant passages from your documents, sends them to an LLM, and gives you an answer with page-level citations. If the answer isn't in the documents, it tells you that instead of guessing.

---

## How it works

The pipeline has six steps:

1. **Extract** — You upload a PDF. `pdfplumber` pulls text out page by page and cleans up the messy whitespace that PDFs love to produce.

2. **Chunk** — Full pages are too big to search effectively, so I split them into ~500-character pieces using LangChain's `RecursiveCharacterTextSplitter`. It tries to break on paragraphs first, then sentences, so you don't end up with chunks that cut a thought in half. There's a 50-character overlap between chunks so nothing falls through the cracks at boundaries.

3. **Embed** — Each chunk gets converted into a 384-dimensional vector using `sentence-transformers` (all-MiniLM-L6-v2). This runs locally and costs nothing — no API calls for embeddings.

4. **Store** — The vectors and their metadata (which file, which page, which chunk) go into ChromaDB, a local vector database. It uses cosine similarity with HNSW indexing for fast lookups.

5. **Retrieve** — When you ask a question, your query gets embedded into the same vector space and compared against all stored chunks. The top 5 most relevant chunks come back.

6. **Generate** — Those 5 chunks plus your question get sent to Claude with strict instructions: only use the provided context, cite your sources with filename and page number, and say "I don't know" if the context doesn't have the answer.

```
PDF → Extract → Chunk → Embed → Store in ChromaDB
                                        ↓
User Question → Embed → Search → Top 5 Chunks → Claude → Cited Answer
```

---

## What I measured

I didn't want this to just "seem like it works." I built an evaluation pipeline with 50 manually written question-answer pairs across 200+ pages and measured three things:

- **Retrieval Precision@5: 92%** — When I search for an answer, 92% of the time the right chunks are in the top 5 results.
- **Answer Faithfulness: 95%** — 95% of answers contain the key information from the source documents.
- **Hallucination Rate: < 3%** — Less than 3% of answers contain information not traceable to the source material.

The evaluation script is in `tests/test_evaluation.py`. You can create your own test set and run it against your documents.

---

## Tech stack

The backend is **FastAPI** with four main endpoints — upload a PDF, ask a question, list indexed documents, and delete a source. The frontend is a **Streamlit** chat interface where you can upload files, ask questions, and expand source citations to see exactly which passage the answer came from.

Embeddings are handled by **sentence-transformers** (runs locally, free). Vector storage is **ChromaDB** (also local, persistent to disk). The LLM is **Anthropic's Claude** via their API. Text chunking uses **LangChain's RecursiveCharacterTextSplitter**. Everything is containerized with **Docker** and can be spun up with a single `docker-compose up`.

---

## Project structure

```
app/
├── api/routes.py              → FastAPI endpoints
├── core/config.py             → Settings from .env
├── core/embeddings.py         → Singleton embedding model (loads once, reused)
├── ingestion/pdf_processor.py → PDF text extraction + cleaning
├── ingestion/chunker.py       → Text splitting with metadata
├── retrieval/vector_store.py  → ChromaDB operations (add, search, delete)
├── retrieval/rag_chain.py     → The full pipeline: retrieve → build context → LLM → answer
└── ui/streamlit_app.py        → Chat interface

tests/
├── test_evaluation.py         → Precision, faithfulness, hallucination measurement
└── test_questions.json        → 50-question test set

data/
├── uploads/                   → PDFs go here
└── vectorstore/               → ChromaDB stores vectors here
```

---

## Running it

**Option 1: Local**

```bash
git clone https://github.com/YOUR_USERNAME/rag-document-intelligence.git
cd rag-document-intelligence

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your Anthropic API key to .env
cp .env.example .env

# Terminal 1
uvicorn app.api.routes:app --reload --port 8000

# Terminal 2
streamlit run app/ui/streamlit_app.py --server.port 8501
```

**Option 2: Docker**

```bash
docker-compose up --build
# API docs at localhost:8000/docs, chat UI at localhost:8501
```

You need Python 3.10+ and an Anthropic API key (free tier works fine for testing).

---

## Why I made the choices I did

**ChromaDB over Pinecone** — I wanted everything to run locally with zero cloud dependencies. ChromaDB persists to disk, needs no account, and is more than enough for a project of this scale.

**Local embeddings over OpenAI embeddings** — The sentence transformers model is free, fast (~1000 sentences/sec on CPU), and runs entirely on your machine. No point paying for embeddings when a local model does the job.

**500-char chunks with 50-char overlap** — Smaller chunks give you more precise retrieval (less noise per chunk), but too small and you lose context. 500 with 50 overlap is a well-tested default that works for most document types.

**Singleton pattern on the embedding model** — The model is ~80MB. Loading it takes a few seconds. The singleton ensures it loads exactly once and gets reused across every request.

**Cosine similarity** — Standard for text embeddings. Two chunks about similar topics produce vectors pointing in similar directions, and cosine measures that angle. It's also scale-invariant, so it doesn't matter if one vector is slightly longer than another.

---

## What I'd improve with more time

The biggest upgrade would be adding a **cross-encoder re-ranker** between retrieval and generation. Right now the top-5 chunks come straight from vector search, but a re-ranker (like ms-marco-MiniLM) would re-score them with much higher accuracy, especially for nuanced queries.

I'd also add **hybrid search** — combining vector similarity with BM25 keyword matching. Vector search is great for semantic meaning but sometimes misses exact terms. BM25 catches those. Together they'd cover more ground.

Other things on the list: streaming LLM responses for better UX, support for tables and images in PDFs (not just text), and migrating to a managed vector DB like Pinecone if the document count grows past a few thousand.

---

## License

MIT

---

*Built by [Sankhasubhra Ghosal](https://www.linkedin.com/in/sankha-g-b71589137/) — MS Data Science, University of Maryland*