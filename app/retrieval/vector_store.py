import chromadb
import os
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from app.core.config import settings
from app.core.embeddings import EmbeddingModel
from app.ingestion.chunker import Chunk



class VectorStore:

    def __init__(self):
        persist_dir = settings.chroma_persist_dir

        try:
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
        except Exception:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = EmbeddingModel()

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.embed_texts(texts)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Stored {len(chunks)} chunks in vector database")
        return len(chunks)

    def search(
        self, query: str, top_k: int = None, source_filter: str = None
    ) -> list[dict]:
        top_k = top_k or settings.top_k

        
        query_embedding = self.embedder.embed_query(query)

        
        where_filter = None
        if source_filter:
            where_filter = {"source_file": source_filter}

        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "text": results["documents"][0][i],
                "source_file": results["metadatas"][0][i]["source_file"],
                "page_number": results["metadatas"][0][i]["page_number"],
                "score": round(1 - results["distances"][0][i], 4),
            })

        return formatted

    def list_sources(self) -> list[str]:
        all_metadata = self.collection.get(include=["metadatas"])
        sources = set()
        for m in all_metadata["metadatas"]:
            sources.add(m["source_file"])
        return sorted(list(sources))

    def get_doc_count(self) -> int:
        return self.collection.count()

    def delete_source(self, source_file: str) -> None:
        self.collection.delete(
            where={"source_file": source_file}
        )
        logger.info(f"Deleted all chunks from {source_file}")