import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from app.core.config import settings
from app.ingestion.pdf_processor import PDFProcessor
from app.ingestion.chunker import DocumentChunker
from app.retrieval.vector_store import VectorStore
from app.retrieval.rag_chain import RAGChain

app = FastAPI(
    title="RAG Document Intelligence API",
    version="1.0.0",
    description="Upload PDFs and ask questions with cited answers",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
chunker = DocumentChunker(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)
vector_store = VectorStore()
rag_chain = RAGChain()


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    question: str
    source_filter: str | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float

class UploadResponse(BaseModel):
    filename: str
    pages_extracted: int
    chunks_created: int
    message: str

class StatsResponse(BaseModel):
    total_chunks: int
    indexed_files: list[str]


# --- Endpoints ---

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    # Save uploaded file
    file_path = os.path.join(settings.upload_dir, file.filename)
    os.makedirs(settings.upload_dir, exist_ok=True)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Saved uploaded file: {file.filename}")

    try:
        # Extract text
        pages = PDFProcessor.extract_pages(file_path)

        # Chunk
        chunks = chunker.chunk_pages(pages)

        # Store in vector DB
        count = vector_store.add_chunks(chunks)

        return UploadResponse(
            filename=file.filename,
            pages_extracted=len(pages),
            chunks_created=count,
            message=f"Successfully indexed {file.filename}",
        )

    except Exception as e:
        logger.error(f"Failed to process {file.filename}: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):

    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    response = rag_chain.query(
        question=request.question,
        source_filter=request.source_filter,
    )

    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        confidence=response.confidence,
    )


@app.get("/sources", response_model=StatsResponse)
async def list_sources():
    return StatsResponse(
        total_chunks=vector_store.get_doc_count(),
        indexed_files=vector_store.list_sources(),
    )


@app.delete("/sources/{filename}")
async def delete_source(filename: str):
    vector_store.delete_source(filename)
    return {"message": f"Deleted {filename} from index"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "chunks_indexed": vector_store.get_doc_count()}