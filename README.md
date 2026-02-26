# RAG Document Intelligence

Upload PDFs. Ask questions. Get cited answers.

I got tired of two things: Ctrl+F not being smart enough for complex questions, and LLMs confidently hallucinating when I asked about my own documents. So I built this.

Drop in any PDF (SEC filings, research papers, policy documents, whatever) and ask questions in plain English. The system finds the exact relevant passages, sends them to an LLM with strict grounding instructions, and gives you an answer with page-level source citations. If the answer isn't in your documents, it says so instead of guessing.

I tested it against Apple's 2024 10-K filing (121 pages) with a 50-question evaluation set that I wrote by hand, covering everything from top-line revenue figures to buried footnotes about Irish tax escrow accounts.

---

## Evaluation Results

| Metric | Score | What it means |
|--------|-------|---------------|
| Hallucination Rate | 0% | Zero answers contained fabricated information |
| Trick Question Detection | 100% (3/3) | Asked about crypto holdings, EV strategy, social media revenue. None exist in the 10-K. System correctly refused all three |
| Answer Faithfulness | 62% | Of real questions, 29/47 contained the expected key terms |
| Retrieval Failures | 38% | 18 questions where the answer was in the PDF but retrieval didn't surface the right chunks |

The system is deliberately conservative. It would rather say "I don't have enough information" than risk making something up. Every single retrieval failure was the system declining to answer, not answering incorrectly. The 0% hallucination rate reflects this design choice.

Most failures were on financial table data (balance sheet items, segment breakdowns) where numbers got split across chunk boundaries during text splitting. This is a known limitation of fixed-size chunking on tabular data and the primary target for improvement.

The evaluation script lives in `tests/test_evaluation.py`, the test set in `tests/test_questions.json`, and raw per-question results in `tests/evaluation_results.csv`.

---

## How It Works

```
PDF Upload > Text Extraction > Chunking > Embedding > ChromaDB
                                                         |
       User Question > Embed Query > Cosine Search > Top-K Chunks > Claude > Cited Answer
```

**Extract.** pdfplumber pulls text page by page. PDFs are messy: inconsistent spacing, merged columns, headers bleeding into body text. The processor cleans this up and tags each page with its source file and page number.

**Chunk.** Full pages are too large for precise retrieval. LangChain's RecursiveCharacterTextSplitter breaks them into roughly 1000-character pieces, trying to split on paragraph boundaries first, then sentences, so chunks don't cut thoughts in half. A 200-character overlap between chunks prevents information loss at boundaries.

**Embed.** Each chunk becomes a 384-dimensional vector using sentence-transformers (all-MiniLM-L6-v2). This runs locally on CPU at about 1000 sentences per second and costs nothing.

**Store.** Vectors and metadata go into ChromaDB with HNSW indexing configured for cosine similarity. Everything persists to disk, so you don't re-embed on restart.

**Retrieve.** Your question gets embedded into the same vector space. ChromaDB returns the top 8 most similar chunks by cosine distance, along with their source file, page number, and similarity score.

**Generate.** The retrieved chunks plus your question go to Claude with strict instructions: answer only from the provided context, cite every claim with the filename and page number, and explicitly state when the context is insufficient. The response includes a confidence score based on average retrieval similarity across all returned chunks.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| PDF Extraction | pdfplumber | Handles messy layouts better than PyPDF2, can extract tables |
| Text Chunking | langchain-text-splitters | Recursive splitting that respects paragraph and sentence boundaries |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, local, fast. 384-dim vectors with no API cost |
| Vector Store | ChromaDB | Local, persistent, zero config. Cosine similarity with HNSW indexing |
| LLM | Anthropic Claude | Strong instruction following for grounded, cited answers |
| REST API | FastAPI | Auto-generated Swagger docs at /docs |
| Frontend | Streamlit | Chat interface with file upload, source citations, confidence display |
| Containerization | Docker + docker-compose | Single command to run everything |

---

## Project Structure

```
app/
    api/routes.py              FastAPI endpoints (upload, query, list sources, delete)
    core/config.py             Central settings, reads .env and Streamlit secrets
    core/embeddings.py         Singleton embedding model, loads once and reuses everywhere
    ingestion/pdf_processor.py PDF to cleaned text with page metadata
    ingestion/chunker.py       Text to overlapping chunks with source tracking
    retrieval/vector_store.py  ChromaDB wrapper (add, search, delete, list)
    retrieval/rag_chain.py     Full pipeline: retrieve, build context, send to Claude, return answer
    ui/streamlit_app.py        Standalone chat UI, calls classes directly without needing the API

tests/
    test_evaluation.py         Retrieval precision, faithfulness, hallucination measurement
    test_questions.json        50 hand-written Q&A pairs with expected answers and page numbers
    evaluation_results.csv     Per-question results from the last evaluation run

data/
    uploads/                   Uploaded PDFs stored here
    vectorstore/               ChromaDB persistence directory
```

---

## Running It

You need Python 3.10+ and an Anthropic API key from console.anthropic.com.

Local setup:
```bash
git clone https://github.com/SankhaS/rag-document-intelligence.git
cd rag-document-intelligence

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env

streamlit run app/ui/streamlit_app.py --server.port 8501
```

Open localhost:8501, upload a PDF, and start asking questions.

With Docker:
```bash
docker-compose up --build
```

If you want the REST API and Swagger docs alongside the chat UI:
```bash
# Terminal 1
uvicorn app.api.routes:app --reload --port 8000

# Terminal 2
streamlit run app/ui/streamlit_app.py --server.port 8501
```

---

## Running the Evaluation

Upload a PDF through the UI first so ChromaDB has something to search. Then:

```bash
python -m tests.test_evaluation
```

This runs all 50 questions, hits Claude 50 times, and prints a scorecard:

```
============================================================
RAG EVALUATION RESULTS
============================================================
Test Questions:         50
Retrieval Precision@5:  XX.XX%
Answer Faithfulness:    XX.XX%
Hallucination Rate:     XX.XX%
No Answer Rate:         XX.XX%
============================================================
Detailed results saved to tests/evaluation_results.csv
```

The test set contains 50 questions against Apple's 2024 10-K, organized by difficulty:

Easy: direct lookups like total revenue, net income, EPS, employee count. Medium: segment breakdowns, product-level revenue, year-over-year comparisons. Hard (financial): balance sheet details, cash flow items, operating margins, tax rates. Hard (reasoning): multi-step questions that require pulling from multiple sections of the document. Hard (qualitative): risk factors, manufacturing locations, competitive landscape. Trick questions: crypto holdings, EV strategy, social media revenue, none of which exist in the 10-K. These test whether the system says "I don't know" instead of hallucinating.

You can write your own test set for any PDF by following the same JSON format.

---

## Design Decisions

**ChromaDB over Pinecone or Weaviate.** Everything runs locally with zero cloud dependencies or accounts. ChromaDB persists to disk and handles the scale this project needs. If the document count grew to tens of thousands, I'd migrate to a managed solution.

**Local embeddings over OpenAI or Cohere.** The all-MiniLM-L6-v2 model is free, runs on CPU, and embeds about 1000 sentences per second. The entire embedding pipeline costs $0. OpenAI embeddings are marginally better but add latency, cost, and an external dependency for no meaningful gain at this scale.

**Singleton pattern on the embedding model.** The model is roughly 80MB. Loading it takes a few seconds. A singleton ensures it loads exactly once and is reused across all requests, whether they come from the API or the Streamlit UI.

**1000-character chunks with 200-character overlap.** Smaller chunks (500) gave more precise retrieval for narrative text but destroyed financial tables. A row of numbers would end up in one chunk while its column headers landed in another. Bumping to 1000 with 200 overlap keeps tables intact without losing too much retrieval precision on prose.

**Cosine similarity.** Standard for normalized text embeddings. Measures the angle between vectors regardless of magnitude, so it doesn't matter if one chunk is slightly longer than another.

**Conservative LLM prompting.** The system prompt tells Claude to refuse to answer if the context is insufficient. This is a deliberate trade-off: lower faithfulness score (the system says "I don't know" more often) in exchange for 0% hallucination. In production, a system that admits uncertainty is more trustworthy than one that always gives an answer.

---

## What I'd Improve

**Cross-encoder re-ranking.** Add a re-ranker like ms-marco-MiniLM between retrieval and generation. Right now the top-K chunks come straight from vector search. A cross-encoder would re-score them with much higher accuracy, especially for nuanced or multi-hop queries.

**Hybrid search.** Combine vector similarity with BM25 keyword matching. Vector search handles semantic meaning well ("What were Apple's earnings?") but can miss exact terms ("What was the 10.2 billion charge?"). BM25 catches lexical matches. Together they'd cover more ground.

**Table-aware chunking.** The biggest weakness right now. Financial tables need special handling, either extracting them separately with pdfplumber's table detection or using a layout-aware chunking strategy that keeps rows and headers together.

**Streaming responses.** Currently the UI waits for the full response before displaying anything. Streaming from Claude's API would show tokens as they arrive, cutting perceived latency significantly.

**Multi-document cross-referencing.** Support queries that span multiple PDFs ("How did Apple's revenue compare to Microsoft's?") by searching across all indexed documents and presenting a synthesized answer.

---

## License

MIT

---

Built by Sankhasubhra Ghosal. MS in Data Science, University of Maryland.

https://www.linkedin.com/in/sankha-g-b71589137/