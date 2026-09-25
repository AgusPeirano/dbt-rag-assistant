"""
Vector store sobre Chroma, en modo embebido (persiste a disco local, no
corre como servidor separado — nada de Docker ni de instalar una base de
datos aparte).

Nota honesta sobre esta elección: originalmente usamos Postgres + pgvector
porque, si venís de Data Engineering, reutiliza algo que ya sabés operar y
se parece más a un stack de producción real. Chroma es la alternativa
"cero fricción" para prototipar: se instala con pip, no necesita un
servicio corriendo aparte, y es exactamente lo que usan un montón de
proyectos RAG reales para iterar rápido. El trade-off es real (Postgres
escala y se opera distinto que una librería embebida) — vale la pena
mencionarlo en el README del proyecto como una decisión consciente, no
esconderlo.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
import numpy as np

from src.chunking import Chunk
from src.config import settings


@dataclass
class RetrievedChunk:
    text: str
    source_path: str
    source_url: str
    header_trail: str
    distance: float  # menor = más parecido (distancia coseno)


class ChromaVectorStore:
    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # misma métrica que usábamos con pgvector
        )

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def insert_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        assert len(chunks) == len(embeddings), "chunks y embeddings deben tener el mismo largo"
        # Chroma necesita un id único por documento; usamos path + índice de chunk
        ids = [f"{c.source_path}::{c.chunk_index}" for c in chunks]
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[
                {"source_path": c.source_path, "source_url": c.source_url, "header_trail": c.header_trail}
                for c in chunks
            ],
        )

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        if not result["ids"] or not result["ids"][0]:
            return []

        out = []
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            out.append(
                RetrievedChunk(
                    text=doc,
                    source_path=meta["source_path"],
                    source_url=meta["source_url"],
                    header_trail=meta.get("header_trail", ""),
                    distance=dist,
                )
            )
        return out

    def count(self) -> int:
        return self.collection.count()
