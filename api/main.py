"""API mínima para exponer el asistente. Correr con:
    uvicorn api.main:app --reload --port 8000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag_pipeline import RAGPipeline

app = FastAPI(title="dbt RAG Assistant", version="0.1.0")
pipeline: RAGPipeline | None = None  # se inicializa en el startup (carga el modelo de embeddings una sola vez)


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceOut(BaseModel):
    source_path: str
    source_url: str
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@app.on_event("startup")
def load_pipeline():
    global pipeline
    pipeline = RAGPipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline no inicializado todavía")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")

    result = pipeline.answer(req.question, top_k=req.top_k)
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceOut(source_path=s.source_path, source_url=s.source_url, distance=s.distance)
            for s in result.sources
        ],
    )
