"""
Ingesta: trae los documentos Markdown de dbt desde GitHub.

Diseño (esto es lo que en un pipeline de datos "de verdad" harías como un
job de extracción):
  1. Un único llamado a la API de árboles de Git (`git/trees?recursive=1`)
     para listar todo el repo en una sola request, en vez de recorrer
     carpeta por carpeta (evita quemar rate limit).
  2. Filtramos a los archivos .md dentro de la carpeta de docs.
  3. Descargamos el contenido real vía raw.githubusercontent.com, que no
     tiene el rate limit agresivo de la API (~60 req/h sin token).

Nota real de la vida: la GitHub API sin autenticar tiene un límite de 60
requests/hora POR IP. Si corrés esto seguido (o desde una IP compartida,
como un sandbox o CI), te vas a quedar sin cupo. Por eso soportamos
GITHUB_TOKEN (subís a 5000 req/h) — generá uno sin permisos especiales en
https://github.com/settings/tokens y ponelo en tu .env.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from src.config import settings


@dataclass
class RawDocument:
    path: str          # ej: "website/docs/docs/introduction.md"
    url: str            # URL de origen (para citar la fuente al usuario final)
    content: str         # contenido markdown crudo


def _github_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


def list_markdown_paths() -> list[str]:
    """Lista todos los .md dentro de GITHUB_DOCS_PATH usando un solo request."""
    url = (
        f"https://api.github.com/repos/{settings.github_repo}"
        f"/git/trees/{settings.github_branch}?recursive=1"
    )
    resp = requests.get(url, headers=_github_headers(), timeout=30)

    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        raise RuntimeError(
            "Te quedaste sin cupo en la API de GitHub (60 req/h sin token). "
            "Configurá GITHUB_TOKEN en tu .env y volvé a intentar. "
            "Ver: https://github.com/settings/tokens"
        )
    resp.raise_for_status()

    tree = resp.json().get("tree", [])
    return [
        item["path"]
        for item in tree
        if item["path"].startswith(settings.github_docs_path)
        and item["path"].endswith(".md")
    ]


def fetch_document(path: str) -> RawDocument:
    """Descarga el contenido crudo de un archivo vía la CDN de GitHub."""
    raw_url = (
        f"https://raw.githubusercontent.com/{settings.github_repo}"
        f"/{settings.github_branch}/{path}"
    )
    resp = requests.get(raw_url, timeout=30)
    resp.raise_for_status()
    return RawDocument(path=path, url=raw_url, content=resp.text)


def fetch_all_documents(limit: int | None = None, delay: float = 0.05) -> list[RawDocument]:
    """Orquesta la ingesta completa. `limit` es útil para pruebas rápidas."""
    paths = list_markdown_paths()
    if limit:
        paths = paths[:limit]

    docs: list[RawDocument] = []
    for i, path in enumerate(paths, start=1):
        try:
            docs.append(fetch_document(path))
        except requests.HTTPError as e:
            print(f"  [WARN] no pude bajar {path}: {e}")
        time.sleep(delay)  # buena práctica: no golpear la CDN sin freno
        if i % 25 == 0:
            print(f"  ...{i}/{len(paths)} documentos descargados")

    print(f"Ingesta completa: {len(docs)} documentos.")
    return docs


if __name__ == "__main__":
    # Prueba rápida: trae solo 5 documentos para validar que todo funciona
    sample = fetch_all_documents(limit=5)
    for d in sample:
        print(f"- {d.path} ({len(d.content)} caracteres)")
