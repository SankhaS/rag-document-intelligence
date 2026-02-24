from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from app.ingestion.pdf_processor import PageContent


@dataclass
class Chunk:
    text: str
    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int

class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_pages(self, pages: list[PageContent]) -> list[Chunk]:
        all_chunks = []

        for page in pages:
            texts = self.splitter.split_text(page.text)

            for idx, text in enumerate(texts):
                chunk = Chunk(
                    text=text,
                    chunk_id=f"{page.source_file}_p{page.page_number}_c{idx}",
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=idx,
                )
                all_chunks.append(chunk)

        logger.info(
            f"Created {len(all_chunks)} chunks from "
            f"{len(pages)} pages "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks