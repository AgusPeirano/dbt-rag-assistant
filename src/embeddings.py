"""
Capa de embeddings, intercambiable por diseño (patrón Strategy).

Por defecto usamos un modelo local (sentence-transformers / bge-small):
gratis, corre en CPU, y no depende de una API externa para la parte que
se ejecuta más veces (cada chunk del corpus). Si más adelante necesitás
mejor calidad semántica o multilingüe, podés cambiar a OpenAI o Voyage AI
(el proveedor de embeddings que suele recomendarse junto con modelos de
Anthropic, ya que Claude no expone un endpoint de embeddings propio)
cambiando una sola variable de entorno: EMBEDDING_PROVIDER.

Importante: si cambiás de modelo de embeddings, tenés que re-indexar todo
el corpus. Los vectores de un modelo no son comparables con los de otro.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.config import settings


class Embedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Devuelve un array (N, dim) de vectores float32."""
        ...


class LocalEmbedder(Embedder):
    """Embeddings 100% locales con sentence-transformers. Sin costo, sin API key."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # import perezoso

        self.model_name = model_name or settings.local_embedding_model
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=len(texts) > 20
        ).astype("float32")


class OpenAIEmbedder(Embedder):
    """Embeddings vía API de OpenAI. Requiere OPENAI_API_KEY."""

    def __init__(self, model_name: str | None = None):
        from openai import OpenAI

        self.model_name = model_name or settings.openai_embedding_model
        self.client = OpenAI(api_key=settings.openai_api_key)

    def encode(self, texts: list[str]) -> np.ndarray:
        # La API de OpenAI acepta batches; 100 por request es un tamaño prudente
        vectors = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            vectors.extend([d.embedding for d in resp.data])
        arr = np.array(vectors, dtype="float32")
        # normalizamos para poder usar similitud coseno consistentemente
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


class VoyageEmbedder(Embedder):
    """Embeddings vía Voyage AI. Requiere VOYAGE_API_KEY. Buena opción si
    querés quedarte en un ecosistema afín a Anthropic para todo el stack.

    Nota: el plan gratis de Voyage SIN tarjeta cargada tiene un límite muy
    bajo (3 requests/min, 10K tokens/min). Por eso acá reintentamos con
    espera en vez de fallar directo ante un RateLimitError — indexar el
    corpus completo tarda más (15-20 min), pero no hace falta pagar nada.
    """

    def __init__(self, model_name: str = "voyage-3.5"):
        import voyageai

        self.model_name = model_name
        self.client = voyageai.Client()

    def encode(self, texts: list[str]) -> np.ndarray:
        import time

        import voyageai

        max_retries = 10
        wait_seconds = 22  # un poco más de 60/3, para no volver a pisar el límite de 3 RPM

        for attempt in range(max_retries):
            try:
                result = self.client.embed(texts, model=self.model_name, input_type="document")
                arr = np.array(result.embeddings, dtype="float32")
                return arr / np.linalg.norm(arr, axis=1, keepdims=True)
            except voyageai.error.RateLimitError:
                if attempt == max_retries - 1:
                    raise
                print(f"  [rate limit de Voyage] esperando {wait_seconds}s antes de reintentar...")
                time.sleep(wait_seconds)

        raise RuntimeError("No se pudo generar el embedding tras varios reintentos")


def get_embedder() -> Embedder:
    provider = settings.embedding_provider.lower()
    if provider == "local":
        return LocalEmbedder()
    if provider == "openai":
        return OpenAIEmbedder()
    if provider == "voyage":
        return VoyageEmbedder()
    raise ValueError(f"EMBEDDING_PROVIDER desconocido: {provider!r}")