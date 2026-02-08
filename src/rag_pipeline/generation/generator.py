from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """You are a helpful RAG assistant.

Answer the question using ONLY the provided context.
Do NOT use any other your internal knowledge.

If the context does not contain the answer, reply exactly:
Insufficient context.

Write a short, direct answer using the context.


"""


class AnswerGenerator:
    def __init__(self, model = "gemma3:4b", temperature= 0.1):
        self.llm = ChatOllama(model=model,temperature=temperature)

    def generate(self, question,retrieved_chunks):

        context = self._build_context(retrieved_chunks)

        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Return a direct answer."

        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)

        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "question": question,
            "answer": answer.strip(),
            "chunks_used": retrieved_chunks,
        }

    @staticmethod
    def _build_context(chunks):
        if not chunks:
            return "No relevant context found."

        blocks = []
        for i, item in enumerate(chunks, start=1):
            meta = item.get("metadata", {})

            source = meta.get("file", meta.get("doc_id", "unknown"))
            page = meta.get("page", "unknown")

            blocks.append(
                f"[chunk_id={i} | source={source} | page={page}]\n"
                f"{item['content']}"
            )

        return "\n\n".join(blocks)

