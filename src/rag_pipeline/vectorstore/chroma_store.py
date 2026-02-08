import chromadb
from chromadb.config import Settings
from pathlib import Path


class ChromaStore:
    def __init__(self, persist_directory, collection_name, embedder):
        """
        embedder: instance of SentenceTransformerEmbedder
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedder = embedder

        self.client = chromadb.Client(
            Settings(persist_directory=str(self.persist_directory))
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks):
        """
        chunks: list of LangChain Document objects
        """
        if not chunks:
            return

        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]

        ids = [
            f"{meta['doc_id']}_{meta.get('page', 1)}_{i}"
            for i, meta in enumerate(metadatas)
        ]

        embeddings = self.embedder.embed_text(chunks)

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )

    def query(self, query_text, n_results=5, where=None):
        """
        where: optional metadata filter
        """
        query_embedding = self.embedder.embed_query(query_text)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        return results
