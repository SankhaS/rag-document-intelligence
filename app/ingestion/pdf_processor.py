import pdfplumber
from pathlib import Path
from loguru import logger
from dataclasses import dataclass

@dataclass
class PageContent:
    text: str
    page_number: int
    source_file: str
    total_pages: int

class PDFProcessor:
    @staticmethod
    def extract_pages( file_path : str) -> list[PageContent]:
        pages = []
        file_name = Path(file_path).name
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processing {file_name}: {total_pages} pages")

                for i,page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        # Clean the extracted text
                        cleaned = PDFProcessor._clean_text(text)

                        pages.append(PageContent(
                            text=cleaned,
                            page_number=i + 1,
                            source_file=file_name,
                            total_pages=total_pages,
                        ))
                    else:
                        logger.warning(
                            f"Page {i+1} of {file_name}: no text extracted"
                        )
        except Exception as e:
            logger.error(f"Failed to process {file_name}: {e}")
            raise

        logger.info(
            f"Extracted {len(pages)} pages with text from {file_name}"
        )
        return pages

    def _clean_text(text: str) -> str:
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()        
    