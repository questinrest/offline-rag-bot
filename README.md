# offline-rag-bot

A lightweight offline RAG (Retrieval-Augmented Generation) pipeline for local/document-based QA and experimentation.

## Architecture
- Entry: [app.py](app.py)
- Ingestion: document loading and splitting via [`rag_pipeline.ingestion.chunker`](src/rag_pipeline/ingestion/chunker.py) and [`rag_pipeline.ingestion.document_loader`](src/rag_pipeline/ingestion/document_loader.py).
- Embedding: embedding creation handled by [`rag_pipeline.embedding.embedding`](src/rag_pipeline/embedding/embedding.py).
- Vector store: persistent vector index (see [src/rag_pipeline/vectorstore](src/rag_pipeline/vectorstore)).
- Retrieval: nearest-neighbor retrieval and scoring (see [src/rag_pipeline/retrieval](src/rag_pipeline/retrieval)).
- Generation: response generation & prompt orchestration in [`rag_pipeline.generation.generator`](src/rag_pipeline/generation/generator.py).

Data flow: documents -> loader -> [`chunker`](src/rag_pipeline/ingestion/chunker.py) -> embeddings (`embedding.py`) -> vectorstore -> retrieval -> generator -> streamlit app (`app.py`).

## Models used
- Embeddings: sentence-transformers (see [requirements.txt](requirements.txt)), e.g., all-MiniLM variant for compact/high-speed embeddings.
- Generation: local/hosted LLMs via Ollama integrations (see [llm/](llm/) and [requirements.txt](requirements.txt)).
- Adjust model choices in the pipeline modules above to swap embeddings or LLMs.

## Install
```sh
python -m venv .venv
.venv/Scripts/activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```


## Run
- Quick run (development):
```sh
streamlit app.py
```
- Typical steps:
  0. Open CMD and run "Ollama Serve" (make sure to pull models before serving (i have used gemma3:1b and gemma3:270m)).
  1. Documents provided in the docs folder, use them to run ingestion in the streamlit interface by selecting chunk_size and chunk_overlap.
  2. After Ingestion, select top_k and and temperature appropriately for querying in the bar to ask question. 



## References
- Code: [app.py](app.py), [`src/rag_pipeline/ingestion/chunker.py`](src/rag_pipeline/ingestion/chunker.py), [`src/rag_pipeline/ingestion/document_loader.py`](src/rag_pipeline/ingestion/document_loader.py), [`src/rag_pipeline/embedding/embedding.py`](src/rag_pipeline/embedding/embedding.py), [`src/rag_pipeline/generation/generator.py`](src/rag_pipeline/generation/generator.py), [src/rag_pipeline/vectorstore](src/rag_pipeline/vectorstore), [src/rag_pipeline/retrieval](src/rag_pipeline/retrieval)
- Dependencies: [requirements.txt](requirements.txt)