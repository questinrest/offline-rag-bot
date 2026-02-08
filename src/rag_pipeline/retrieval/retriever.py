from typing import List, Dict, Any, Optional
from src.rag_pipeline.embedding.embedding import SentenceTransformerEmbedder
from src.rag_pipeline.vectorstore.chroma_store import ChromaStore


class Retriever:
    def __init__(self, embedder, store, top_k = 5):

        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, question, filters = None):
        """
        returns:
        [
            {
                "content": str,
                "metadata": dict,
                "score": float
            }
        ]
        """
        if not question.strip():
            return []

        results = self.store.query(
            query_text=question,
            n_results=self.top_k,
            where=filters,
        )

        return self._format_results(results)

    @staticmethod
    def _format_results(results):
        """
        Convert Chroma raw output into a clean structure.
        """
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            formatted.append(
                {
                    "content": doc,
                    "metadata": meta,
                    "score": 1 - dist,
                }
            )

        return formatted
