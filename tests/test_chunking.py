import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_document, _strip_mdx_noise, _strip_front_matter
from src.ingestion import RawDocument

SAMPLE_MDX = """---
title: "Ejemplo"
id: "ejemplo"
---
import Foo from '/snippets/_foo.md';

# Sección uno

<Foo />

Contenido real de la sección uno.

## Subsección

Más contenido, esta vez en una subsección anidada.

# Sección dos

Contenido de otra sección top-level.
"""


def test_strip_front_matter_removes_yaml_header():
    cleaned = _strip_front_matter(SAMPLE_MDX)
    assert "title:" not in cleaned
    assert "import Foo" in cleaned  # todavía no limpiamos MDX acá


def test_strip_mdx_noise_removes_imports_and_self_closing_tags():
    cleaned = _strip_mdx_noise(SAMPLE_MDX)
    assert "import Foo" not in cleaned
    assert "<Foo />" not in cleaned
    assert "Contenido real de la sección uno." in cleaned


def test_chunk_document_preserves_header_trail():
    doc = RawDocument(path="test.md", url="http://example.com/test.md", content=SAMPLE_MDX)
    chunks = chunk_document(doc)

    assert len(chunks) >= 3
    subsection_chunk = next(c for c in chunks if "Subsección" in c.header_trail)
    assert subsection_chunk.header_trail == "Sección uno > Subsección"
    assert "Más contenido" in subsection_chunk.text


def test_chunk_document_handles_empty_content():
    doc = RawDocument(path="empty.md", url="http://example.com/empty.md", content="")
    assert chunk_document(doc) == []


def test_long_section_is_split_with_overlap():
    from src.chunking import _split_long_section

    text = "x" * 2000
    parts = _split_long_section(text, size=800, overlap=100)
    assert len(parts) > 1
    # el final del primer chunk debe solaparse con el inicio del segundo
    assert parts[0][-100:] == parts[1][:100]
