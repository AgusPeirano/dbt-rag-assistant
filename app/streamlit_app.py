"""UI de chat mínima. Correr con: streamlit run app/streamlit_app.py"""
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_pipeline import RAGPipeline
from scripts.build_index import build_index

st.set_page_config(page_title="dbt RAG Assistant", page_icon="🔧")
st.title("🔧 Asistente de dbt (RAG)")
st.caption("Respuestas generadas a partir de la documentación oficial de dbt.")


@st.cache_resource
def load_pipeline():
    pipeline = RAGPipeline()

    # Si el índice está vacío (por ejemplo, primer arranque en un hosting
    # con disco efímero como Streamlit Cloud, donde data/chroma_db/ no
    # persiste entre reinicios), lo armamos solos antes de aceptar preguntas.
    if pipeline.store.count() == 0:
        limit_env = os.getenv("STARTUP_INDEX_LIMIT")
        limit = int(limit_env) if limit_env else None
        status = st.empty()

        def report(msg: str):
            status.info(msg)

        with st.spinner("Preparando el índice por primera vez (puede tardar unos minutos)..."):
            build_index(limit=limit, progress_cb=report)
        status.empty()

    return pipeline


if "history" not in st.session_state:
    st.session_state.history = []

pipeline = load_pipeline()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Preguntá algo sobre dbt...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la documentación..."):
            result = pipeline.answer(question)
        st.markdown(result.answer)
        if result.sources:
            with st.expander(f"📚 {len(result.sources)} fuentes consultadas"):
                for s in result.sources:
                    st.markdown(f"- `{s.source_path}` (distancia: {s.distance:.3f})")

    st.session_state.history.append({"role": "assistant", "content": result.answer})
