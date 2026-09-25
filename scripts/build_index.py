"""
Job de indexado end-to-end. Pensalo como el "DAG" del proyecto (en una
versión más madura, esto sería justamente un DAG de Airflow con una tarea
por etapa: ingest -> chunk -> embed -> load, cada una con sus reintentos
y su propio SLA).

Uso:
    python scripts/build_index.py                 # corpus completo
    python scripts/build_index.py --limit 20       # prueba rápida, 20 docs
    python scripts/build_index.py --rebuild         # borra el índice y lo arma de nuevo
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_documents
from src.embeddings import get_embedder
from src.ingestion import fetch_all_documents
from src.vectorstore import ChromaVectorStore


def build_index(limit: int | None = None, rebuild: bool = False, batch_size: int = 64, progress_cb=None) -> int:
    """La lógica de indexado en sí, separada de la CLI para poder llamarla
    también desde la app (por ejemplo, para auto-indexar en el primer
    arranque en un hosting con disco efímero como Streamlit Cloud).

    progress_cb, si se pasa, es una función que recibe un string por cada
    etapa — así la UI puede mostrar el progreso en vez de solo verlo en
    una terminal que el usuario final no tiene.
    """
    def log(msg: str):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    t0 = time.time()

    log("1/4 · Ingesta desde GitHub...")
    docs = fetch_all_documents(limit=limit)

    log("2/4 · Chunking...")
    chunks = chunk_documents(docs)

    log("3/4 · Generando embeddings...")
    embedder = get_embedder()
    store = ChromaVectorStore()
    if rebuild:
        store.clear()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embedder.encode([c.text for c in batch])
        store.insert_chunks(batch, vectors)
        log(f"  ...{min(i + batch_size, len(chunks))}/{len(chunks)} chunks indexados")

    log("4/4 · Listo.")
    log(f"Total en el índice: {store.count()} chunks")
    log(f"Tiempo total: {time.time() - t0:.1f}s")
    return store.count()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="limitar cantidad de docs (para pruebas)")
    parser.add_argument("--rebuild", action="store_true", help="vaciar el índice antes de cargar")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    build_index(limit=args.limit, rebuild=args.rebuild, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
