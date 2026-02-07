from typing import List, Dict, Any
from langchain_ollama import OllamaLLM


SYSTEM_PROMPT = """You are a documentation assistant who specializes into explaining concept topics in easy language by elaborating.

Rules:
- Answer ONLY using the provided context.
- If the relevant context is not present, say "I don't know".
- Do NOT add external knowledge.
- Write COMPLETE sentences.
- Provide a clear definition or explanation.
- Answer in 2–4 sentences.
"""


class AnswerGenerator:
    def __init__(self, model: str = "gemma3:270m", temperature: float = 0.1):
        self.llm = OllamaLLM(model=model,temperature=temperature)

    def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]):

        context = self._build_context(retrieved_chunks)

        prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

        return {
            "question": question,
            "answer": answer.strip(),
            "chunks_used": retrieved_chunks,
        }

    @staticmethod
    def _build_context(chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant context found."

        blocks = []
        for item in chunks[:3]:
            blocks.append(item["content"])

        return "\n\n".join(blocks)
