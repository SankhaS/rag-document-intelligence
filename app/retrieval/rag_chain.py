import anthropic
from loguru import logger
from app.core.config import settings
from app.retrieval.vector_store import VectorStore
from dataclasses import dataclass

@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]  
    confidence: float

SYSTEM_PROMPT = """You are a precise document analysis assistant.
You answer questions ONLY based on the provided context from the
documents. Follow these rules strictly:

1. Only use information from the provided CONTEXT sections below.
2. If the context doesn't contain enough information to answer,
   say "I don't have enough information in the provided documents
   to answer this question."
3. After your answer, cite your sources using the format:
   [Source: filename, Page X]
4. Be concise but thorough. Use bullet points for complex answers.
5. Never make up information or use knowledge outside the context.
"""

class RAGChain:
    def __init__(self):
        self.vector_store = VectorStore()
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )
    
    def query(self, question: str, source_filter: str = None) -> RAGResponse:
        logger.info("Question: {question}")
        results = self.vector_store.search(
            query=question,
            top_k=settings.top_k,
            source_filter=source_filter,
        )
    
        if not results:
            return RAGResponse(
                answer="No documents have been indexed yet. "
                       "Please upload a PDF first.",
                sources=[],
                confidence=0.0,
            )
        
        context = self._build_context(results)
        answer = self._generate_answer(question, context)
        avg_score = sum(r["score"] for r in results) / len(results)

        return RAGResponse(
            answer=answer,
            sources=results,
            confidence=round(avg_score, 4),
        )
    
    def _build_context(self, results: list[dict]) -> str:
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"--- CONTEXT {i} ---\n"
                f"Source: {r['source_file']}, Page {r['page_number']}\n"
                f"Relevance Score: {r['score']}\n"
                f"Content:\n{r['text']}\n"
            )
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:

        user_message = (
            f"CONTEXT FROM DOCUMENTS:\n\n{context}\n\n"
            f"---\n\n"
            f"QUESTION: {question}\n\n"
            f"Please answer based ONLY on the context above. "
            f"Cite sources using [Source: filename, Page X] format."
        )

        try:
            response = self.client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_message}
                ],
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {str(e)}"