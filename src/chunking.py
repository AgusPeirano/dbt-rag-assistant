"""
Chunking consciente de la estructura Markdown.

Por qué no partimos "a lo bruto" cada N caracteres: cortar en medio de una
sección rompe el contexto semántico (el chunk pierde el título que lo
explica) y degrada la calidad del retrieval. En cambio:

  1. Quitamos el front-matter YAML (metadata de Docusaurus, no es contenido).
  2. Partimos primero por headers (#, ##, ###) para respetar la estructura
     lógica del documento.
  3. Si una sección resultante es más larga que CHUNK_SIZE, la partimos en
     sub-chunks con overlap (para no perder contexto en los bordes).
  4. Cada chunk se guarda junto con el trail de headers al que pertenece
     (ej. "Introducción > Instalación > Requisitos"), y se lo antepone al
     texto — esto mejora mucho la calidad del embedding, porque el chunk
     queda auto-contenido en vez de depender de contexto externo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import settings
from src.ingestion import RawDocument

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
FRONT_MATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
MDX_IMPORT_RE = re.compile(r"^import\s+.*?;?\s*$", re.MULTILINE)
MDX_SELF_CLOSING_TAG_RE = re.compile(r"^\s*<[A-Z][A-Za-z0-9]*\s*/>\s*$", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    source_path: str
    source_url: str
    header_trail: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _strip_front_matter(text: str) -> str:
    return FRONT_MATTER_RE.sub("", text, count=1)


def _strip_mdx_noise(text: str) -> str:
    """Los docs de dbt están en MDX (Docusaurus), no Markdown puro: tienen
    imports de componentes React y tags auto-cerrados (ej. <DbtFramework />)
    que no aportan contenido semántico y solo agregan ruido al embedding.
    Esto es dato sucio real, no un caso de laboratorio."""
    text = MDX_IMPORT_RE.sub("", text)
    text = MDX_SELF_CLOSING_TAG_RE.sub("", text)
    return text


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Devuelve [(header_trail, contenido_de_la_seccion), ...]."""
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return [("", text.strip())] if text.strip() else []

    sections: list[tuple[str, str]] = []
    trail: list[tuple[int, str]] = []  # (nivel, titulo)

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Mantenemos el trail de headers "activos" en cada nivel
        trail = [t for t in trail if t[0] < level] + [(level, title)]
        header_trail = " > ".join(t[1] for t in trail)

        if body:
            sections.append((header_trail, body))

    return sections


def _split_long_section(text: str, size: int, overlap: int) -> list[str]:
    """Ventaneo simple con overlap para secciones que exceden chunk_size."""
    if len(text) <= size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap  # el overlap evita cortar ideas a la mitad
    return chunks


def chunk_document(doc: RawDocument) -> list[Chunk]:
    text = _strip_front_matter(doc.content)
    text = _strip_mdx_noise(text)
    sections = _split_by_headers(text)

    chunks: list[Chunk] = []
    idx = 0
    for header_trail, body in sections:
        for piece in _split_long_section(body, settings.chunk_size, settings.chunk_overlap):
            # Anteponer el trail de headers ancla el chunk semánticamente
            enriched_text = f"{header_trail}\n\n{piece}" if header_trail else piece
            chunks.append(
                Chunk(
                    text=enriched_text.strip(),
                    source_path=doc.path,
                    source_url=doc.url,
                    header_trail=header_trail,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks


def chunk_documents(docs: list[RawDocument]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    print(f"Chunking completo: {len(docs)} docs -> {len(all_chunks)} chunks.")
    return all_chunks
