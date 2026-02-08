# offline-rag-bot

A **lightweight, fully offline Retrieval-Augmented Generation (RAG) pipeline** for local document-based question answering and experimentation using **local embeddings, a persistent vector store, and locally served LLMs via Ollama**.

This project is designed to be **simple, modular, and hackable**, making it ideal for learning RAG internals or quickly testing different models and chunking strategies.

---

## Architecture

### High-Level RAG Flow

Documents → Loader → Chunker → Embeddings → Vector Store → Retrieval → LLM → Streamlit UI


The pipeline follows the standard **Retrieval-Augmented Generation (RAG)** workflow:

1. Load documents (PDF / TXT)
2. Split documents into chunks
3. Embed chunks into vector representations
4. Store embeddings in a persistent vector database
5. Retrieve relevant chunks for a user query
6. Generate an answer using an LLM with retrieved context

---

### Components Overview

#### 1. Document Ingestion
- Custom **PDF and TXT extractors**
- Supports:
  - Single file
  - Multiple files
  - Entire folder ingestion
- Converts extracted content into **LangChain `Document` objects**

Relevant modules:
- `rag_pipeline.ingestion.document_loader`
- `rag_pipeline.ingestion.chunker`

---

#### 2. Chunking Strategy
- Uses **recursive text splitting**
- User-configurable:
  - `chunk_size`
  - `chunk_overlap`
- Chunking parameters can be selected directly from the Streamlit UI
- Enables experimentation with different chunking combinations

---

#### 3. Embedding Layer
- Converts text chunks into dense vector embeddings
- Embeddings are generated locally using Sentence Transformers

Module:
- `rag_pipeline.embedding.embedding`

---

#### 4. Vector Store
- Uses **ChromaDB**
- Features:
  - Persistent on-disk storage (not in-memory)
  - **HNSW indexing**
  - **Cosine similarity** search
- Stored embeddings are reused across sessions

Modules:
- `rag_pipeline.vectorstore`
- `rag_pipeline.retrieval`

---

#### 5. Retrieval
- Performs nearest-neighbor search on embedded chunks
- Configurable `top_k`
- Returns:
  - Relevant document chunks
  - Similarity scores

---

#### 6. Generation
- Uses locally served LLMs via **Ollama**
- Handles prompt orchestration and response generation

Module:
- `rag_pipeline.generation.generator`

---

#### 7. User Interface
- Built using **Streamlit**
- Allows:
  - Document ingestion
  - Chunking configuration
  - Model selection
  - Querying with temperature and top-k control
- Displays:
  - Generated answers
  - Retrieved chunks with similarity scores

Entry point:
- `app.py`

---

## Models Used

### Embeddings
- **all-MiniLM-L6-v2**
  - From `sentence-transformers`
  - Compact, fast, and well-suited for local RAG pipelines

---

### LLMs (via Ollama)
Tested models include:

- **Gemma**
  - `gemma3:270m`
  - `gemma3:1b`
  - `gemma3:4b`
- **Qwen**
  - `qwen2.5:3b`
- **Phi**
  - `phi3:mini`
  - `phi3:8b`
- **LLaMA**
  - `llama3.2:1b`
  - `llama3.2:3b`

> Any Ollama-supported model can be added by updating the model list in `app.py`.

---

### Vector Database
- **ChromaDB**
  - Persistent directory-based storage
  - HNSW indexing
  - Cosine similarity search

---

# Offline RAG Bot

An offline Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit** and **Ollama**, supporting persistent embeddings and multiple document formats.

---

## How to Run?

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd offline-rag-bot
```

### 2. Create and Activate enviroment

Windows
```sh
py -3.11 -m venv .venv
.venv\Scripts\activate
```

macOS / Linux
```sh
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Install Ollama

Download and install Ollama
https://ollama.com

```sh
Pull at least one model:
ollama pull gemma3:270m
ollama pull gemma3:1b
ollama pull gemma3:4b
```

### Running the Application
#### 1. Start Ollama
Open a new terminal / CMD and run:
```sh
ollama serve
```

#### 2. Run the Streamlit App
In another terminal (with the virtual environment activated):
```sh
streamlit run app.py
```

### Using the App

#### Step 1: Ingest Documents
Place documents inside the docs/ folder

In the Streamlit UI:

Select document type (PDF or TXT)

Choose chunk_size and chunk_overlap

Click Ingest

Embeddings are stored persistently, so ingestion does not need to be repeated every run.

#### Step 2: Configure Query Settings
Select an LLM from the dropdown

Configure:

top_k (number of retrieved chunks)

temperature

To use a new Ollama model, add it to the model list in app.py:
```sh
model_name = st.sidebar.selectbox(...)
```

#### Step 3: Ask Questions
Type your query in the input box

Press Enter

The app returns:

Generated answer

Retrieved document chunks

Similarity scores
