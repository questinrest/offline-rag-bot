import streamlit as st
import tempfile
from pathlib import Path

from src.rag_pipeline.ingestion.document_loader import to_langchain_docs
from src.rag_pipeline.ingestion.chunker import chunker
from src.rag_pipeline.embedding.embedding import SentenceTransformerEmbedder
from src.rag_pipeline.vectorstore.chroma_store import ChromaStore
from src.rag_pipeline.retrieval.retriever import Retriever
from src.rag_pipeline.generation.generator import AnswerGenerator
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


CHROMA_DIR = "chroma_db"


COLLECTION_NAME = "privacy_docs"


st.set_page_config(page_title="RAG Legal Assistant", layout="wide")
st.title("📄 Legal RAG Assistant (Ollama)")

# ---------------------------
# Sidebar – Ingestion controls
# ---------------------------
#st.sidebar.header("Choose Vector Store or Create One")

#COLLECTION_NAME = st.sidebar.chat_input("Your collection name")

st.sidebar.header("Ingestion")


upload_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf","txt"],
    accept_multiple_files=True,
)

folder_path = st.sidebar.text_input(
    "Or provide a folder path",
    placeholder="folder-path",
)

chunk_size = st.sidebar.slider("Chunk size", 150, 600, 250)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 150, 40)

ingest_button = st.sidebar.button("Run ingestion")


# Sidebar, Retrieval 

st.sidebar.header("Retrieval")

top_k = st.sidebar.slider("Top K", 1, 12, 3)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1)

model_name = st.sidebar.selectbox(
    "Ollama model",
    ["gemma3:270m", "gemma3:1b"],
)


# Initialize components (cached)
@st.cache_resource
def load_embedder():
    return SentenceTransformerEmbedder("all-MiniLM-L6-v2")


@st.cache_resource
def load_store(_embedder):
    return ChromaStore(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedder=_embedder,
    )



embedder = load_embedder()
store = load_store(embedder)

# Ingestion logic

if ingest_button:
    with st.spinner("Ingesting documents..."):
        paths = []

        if upload_files:
            tmp_dir = tempfile.TemporaryDirectory()
            for f in upload_files:
                tmp_path = Path(tmp_dir.name) / f.name
                tmp_path.write_bytes(f.read())
                paths.append(tmp_path)

        elif folder_path:
            paths.append(Path(folder_path))

        else:
            st.warning("Please upload files or provide a folder path.")

        all_docs = []
        for p in paths:
            all_docs.extend(to_langchain_docs(p))

        chunks = chunker(
            all_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        store.add_documents(chunks)

        st.success(f"Ingested {len(chunks)} chunks")


# Question answering

st.header("Ask a question")

question = st.text_input(
    "Enter your question",
    placeholder="Ask Your Question with respect to the document uploaded...",
)

if question:
    retriever = Retriever(
        embedder=embedder,
        store=store,
        top_k=top_k,
    )

    generator = AnswerGenerator(
        model=model_name,
        temperature=temperature,
    )

    with st.spinner("Thinking..."):
        chunks = retriever.retrieve(question)

        if not chunks:
            st.warning("No relevant context found.")
        else:
            result = generator.generate(question, chunks)

            st.subheader("Answer")
            st.write(result["answer"])

            with st.expander("Retrieved Context"):
                for c in chunks:
                    meta = c["metadata"]
                    st.markdown(
                        f"**{meta.get('source')} – page {meta.get('page')}** "
                        f"(score: {c['score']:.3f})"
                    )
                    st.write(c["content"])
                    st.divider()
