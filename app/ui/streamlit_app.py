import sys, os
sys.path.insert(0, os.getcwd())

import streamlit as st
import time

from app.core.config import settings
from app.ingestion.pdf_processor import PDFProcessor
from app.ingestion.chunker import DocumentChunker
from app.retrieval.vector_store import VectorStore
from app.retrieval.rag_chain import RAGChain
from app.core.config import settings

st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📁",
    layout="wide",
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
    st.session_state.chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    st.session_state.rag_chain = RAGChain()
    st.session_state.messages = []

vector_store = st.session_state.vector_store
chunker = st.session_state.chunker
rag_chain = st.session_state.rag_chain

# --- Header ---
st.title("📁 RAG Document Intelligence")
st.caption(
    "Upload PDFs and ask questions, the answers are grounded "
    "in your documents with citations."
)

# --- Sidebar: Upload & Stats ---
with st.sidebar:
    st.header("📁 Document Management")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Indexing document..."):
            # Save uploaded file to disk
            os.makedirs(settings.upload_dir, exist_ok=True)
            file_path = os.path.join(settings.upload_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Extract text
                pages = PDFProcessor.extract_pages(file_path)

                # Chunk
                chunks = chunker.chunk_pages(pages)

                # Store in vector DB
                count = vector_store.add_chunks(chunks)

                st.success(
                    f"✅ **{uploaded_file.name}**\n\n"
                    f"Pages: {len(pages)} | Chunks: {count}"
                )
            except Exception as e:
                st.error(f"Failed to process: {e}")

    st.divider()
    st.header("📊 Index Stats")

    total_chunks = vector_store.get_doc_count()
    indexed_files = vector_store.list_sources()

    st.metric("Total Chunks", total_chunks)

    if indexed_files:
        st.write("**Indexed Files:**")
        for f in indexed_files:
            col1, col2 = st.columns([3, 1])
            col1.write(f"📄 {f}")
            if col2.button("🗑️", key=f"del_{f}"):
                vector_store.delete_source(f)
                st.rerun()
    else:
        st.info("No documents indexed yet. Upload a PDF!")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 View Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**{s['source_file']}** — Page {s['page_number']} "
                        f"(relevance: {s['score']:.2%})"
                    )
                    st.caption(s["text"][:300] + "...")
                    st.divider()


if question := st.chat_input("Ask a question about your documents..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)


    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            start = time.time()

            response = rag_chain.query(question=question)

            elapsed = time.time() - start

            st.markdown(response.answer)

            st.caption(
                f"⏱️ {elapsed:.1f}s | "
                f"Confidence: {response.confidence:.2%}"
            )

            # Show sources
            if response.sources:
                with st.expander("📚 View Sources"):
                    for s in response.sources:
                        st.markdown(
                            f"**{s['source_file']}** — "
                            f"Page {s['page_number']} "
                            f"(relevance: {s['score']:.2%})"
                        )
                        st.caption(s["text"][:300] + "...")
                        st.divider()

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
            })