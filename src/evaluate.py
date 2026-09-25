"""
Evaluación con el "RAG triad" — las tres preguntas que definen si un
sistema RAG anda bien o solo parece andar bien en la demo:

  1. Context Relevance:  ¿los chunks recuperados son relevantes a la pregunta?
  2. Groundedness:       ¿la respuesta generada está sustentada en ese contexto,
                         o el LLM se mandó algo por su cuenta?
  3. Answer Relevance:   ¿la respuesta realmente contesta lo que se preguntó?

Cada métrica se calcula con el propio LLM como juez (LLM-as-judge), un
patrón estándar cuando no tenés un dataset de referencia con respuestas
"correctas" etiquetadas a mano. Para un proyecto más maduro, esto se
reemplaza (o se complementa) con la librería `ragas`, que implementa estas
mismas ideas de forma más rigurosa sobre un dataset de evaluación curado.

Uso:
    python -m src.evaluate --questions data/eval_questions.txt
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass

from src.rag_pipeline import RAGPipeline, _call_anthropic
from src.config import settings

JUDGE_PROMPT = """Evaluá la siguiente interacción de un sistema RAG en tres \
dimensiones, cada una de 1 (muy malo) a 5 (excelente). Respondé SOLO con un \
JSON, sin texto adicional, con este formato exacto:
{{"context_relevance": <1-5>, "groundedness": <1-5>, "answer_relevance": <1-5>, "reasoning": "<una oracion breve>"}}

PREGUNTA: {question}

CONTEXTO RECUPERADO:
{context}

RESPUESTA GENERADA:
{answer}

Definiciones:
- context_relevance: qué tan relevante es el contexto recuperado para la pregunta.
- groundedness: qué tan sustentada está la respuesta en el contexto (5 = todo verificable en el contexto, 1 = inventado).
- answer_relevance: qué tan bien la respuesta contesta lo que se preguntó."""


@dataclass
class EvalResult:
    question: str
    context_relevance: int
    groundedness: int
    answer_relevance: int
    reasoning: str


def evaluate_single(pipeline: RAGPipeline, question: str) -> EvalResult:
    result = pipeline.answer(question)
    context_text = "\n---\n".join(s.text for s in result.sources)

    judge_input = JUDGE_PROMPT.format(question=question, context=context_text, answer=result.answer)
    raw = _call_anthropic("Sos un evaluador estricto y objetivo de sistemas RAG.", judge_input)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data = json.loads(match.group(0)) if match else {}

    return EvalResult(
        question=question,
        context_relevance=data.get("context_relevance", 0),
        groundedness=data.get("groundedness", 0),
        answer_relevance=data.get("answer_relevance", 0),
        reasoning=data.get("reasoning", ""),
    )


def run_evaluation(questions: list[str]) -> list[EvalResult]:
    pipeline = RAGPipeline()
    results = [evaluate_single(pipeline, q) for q in questions]

    if results:
        avg = lambda field: sum(getattr(r, field) for r in results) / len(results)
        print("\n=== Resumen ===")
        print(f"Context relevance promedio: {avg('context_relevance'):.2f}/5")
        print(f"Groundedness promedio:      {avg('groundedness'):.2f}/5")
        print(f"Answer relevance promedio:  {avg('answer_relevance'):.2f}/5")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, help="archivo .txt con una pregunta por línea")
    args = parser.parse_args()

    if args.questions:
        with open(args.questions) as f:
            qs = [line.strip() for line in f if line.strip()]
    else:
        qs = [
            "¿Qué es un modelo incremental en dbt?",
            "¿Cómo defino un test personalizado en dbt?",
            "¿Qué diferencia hay entre dbt Core y dbt Cloud?",
        ]

    for r in run_evaluation(qs):
        print(f"\nQ: {r.question}")
        print(f"  context_relevance={r.context_relevance} groundedness={r.groundedness} answer_relevance={r.answer_relevance}")
        print(f"  → {r.reasoning}")
