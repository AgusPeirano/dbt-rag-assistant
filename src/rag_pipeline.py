"""
El corazón del sistema: dada una pregunta del usuario, recupera contexto
relevante y genera una respuesta que cita sus fuentes.

Nota de diseño: el LLM solo puede responder con lo que está en el
`context` que le pasamos armado — el prompt se lo deja explícito. Esto es
lo que reduce alucinaciones respecto de preguntarle directo al modelo:
si no hay contexto relevante, tiene que decir que no sabe, no inventar.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.embeddings import get_embedder
from src.vectorstore import ChromaVectorStore, RetrievedChunk

SYSTEM_PROMPT = """Sos un asistente experto en dbt (data build tool). \
Respondé la pregunta del usuario ÚNICAMENTE con la información del \
CONTEXTO provisto abajo. Si el contexto no alcanza para responder, decí \
explícitamente que no tenés información suficiente en la documentación \
indexada — no inventes nada. Citá de qué documento sacaste cada afirmación \
usando el path que aparece junto a cada fragmento de contexto."""


@dataclass
class RAGAnswer:
    answer: str
    sources: list[RetrievedChunk]


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Fuente {i}: {c.source_path}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def _call_anthropic(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=800,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _call_openai(system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=800,
    )
    return resp.choices[0].message.content


class RAGPipeline:
    def __init__(self):
        self.embedder = get_embedder()
        self.store = ChromaVectorStore()

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        top_k = top_k or settings.top_k

        query_vec = self.embedder.encode([question])[0]
        retrieved = self.store.similarity_search(query_vec, top_k=top_k)

        if not retrieved:
            return RAGAnswer(
                answer="No hay nada indexado todavía — corré scripts/build_index.py primero.",
                sources=[],
            )

        context = _build_context(retrieved)
        user_prompt = f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}"

        if settings.llm_provider == "anthropic":
            answer_text = _call_anthropic(SYSTEM_PROMPT, user_prompt)
        elif settings.llm_provider == "openai":
            answer_text = _call_openai(SYSTEM_PROMPT, user_prompt)
        else:
            raise ValueError(f"LLM_PROVIDER desconocido: {settings.llm_provider!r}")

        return RAGAnswer(answer=answer_text, sources=retrieved)


if __name__ == "__main__":
    pipeline = RAGPipeline()
    result = pipeline.answer("¿Qué es un modelo incremental en dbt?")
    print(result.answer)
    print("\nFuentes:")
    for s in result.sources:
        print(f"  - {s.source_path} (distancia: {s.distance:.4f})")
