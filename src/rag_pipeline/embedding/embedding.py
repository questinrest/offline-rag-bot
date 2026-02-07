from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    def __init__(self, model):
        self.model_name = model
        self.model = SentenceTransformer(
            model,
            local_files_only=True,
        )

    def embed_text(self, chunks):
        if not chunks:
            return []

        texts = [chunk.page_content for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
        )
        return embedding.tolist()
