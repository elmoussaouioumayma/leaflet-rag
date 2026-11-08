"""FastAPI service with health check, Prometheus metrics and structured JSON logs.

Privacy: logs never contain the question text, only a short hash, so questions about
someone's health are not stored in plain text.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from . import __version__
from .pipeline import LeafletAssistant

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("leaflet_rag.api")

REQUESTS = Counter("leaflet_requests_total", "Questions answered, by outcome status", ["status"])
LATENCY = Histogram("leaflet_latency_seconds", "End-to-end answer latency")
TOP_SCORE = Histogram(
    "leaflet_top_retrieval_score",
    "Best retrieval score per question (a falling distribution signals drift or missing leaflets)",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
)


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["assistant"] = LeafletAssistant.from_index()
    yield
    _state.clear()


app = FastAPI(title="Medicine Leaflet Assistant", version=__version__, lifespan=lifespan)


def get_assistant() -> LeafletAssistant:
    assistant = _state.get("assistant")
    if assistant is None:
        raise HTTPException(503, "Index not loaded. Run `python -m leaflet_rag ingest` first.")
    return assistant


@app.get("/health")
def health() -> dict:
    assistant = _state.get("assistant")
    return {
        "status": "ok" if assistant else "no_index",
        "version": __version__,
        "chunks": len(assistant.retriever.chunks) if assistant else 0,
        "retriever": assistant.retriever.backend if assistant else None,
        "generator": assistant.generator.name if assistant else None,
    }


@app.post("/ask")
def ask(payload: Question) -> dict:
    answer = get_assistant().ask(payload.question)
    REQUESTS.labels(status=answer.status).inc()
    LATENCY.observe(answer.latency_ms / 1000)
    if answer.status in {"answered", "not_found"}:
        TOP_SCORE.observe(answer.top_score)
    logger.info(
        json.dumps(
            {
                "event": "ask",
                "question_hash": hashlib.sha256(payload.question.encode()).hexdigest()[:12],
                "status": answer.status,
                "generator": answer.generator,
                "top_score": answer.top_score,
                "n_citations": len(answer.citations),
                "latency_ms": answer.latency_ms,
            }
        )
    )
    return asdict(answer)


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
