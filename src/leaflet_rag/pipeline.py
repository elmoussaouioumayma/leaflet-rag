"""End-to-end question answering: guardrails -> retrieval -> generation -> citation check."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from .config import Settings, get_settings
from .generator import NOT_FOUND, ExtractiveGenerator, LLMGenerator, cited_numbers
from .guardrails import check_question
from .retriever import Hit, Retriever, load_retriever

logger = logging.getLogger("leaflet_rag")

DISCLAIMER = "Information from package leaflets only. It is not medical advice; ask a pharmacist or doctor."
NO_ANSWER = "I could not find this in the available package leaflets. Please ask a pharmacist."


@dataclass
class Citation:
    number: int
    doc: str
    section: str
    section_title: str
    score: float
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    status: str  # "answered", "not_found", "refused_emergency", "refused_personal_advice"
    citations: list[Citation] = field(default_factory=list)
    generator: str = ""
    top_score: float = 0.0
    latency_ms: float = 0.0
    disclaimer: str = DISCLAIMER


class LeafletAssistant:
    def __init__(self, retriever: Retriever, settings: Settings | None = None, generator=None):
        self.settings = settings or get_settings()
        self.retriever = retriever
        self.fallback = ExtractiveGenerator()
        if generator is not None:
            self.generator = generator
        elif self.settings.llm_enabled:
            s = self.settings
            self.generator = LLMGenerator(s.llm_model, s.llm_base_url, s.llm_api_key, s.llm_temperature)
        else:
            self.generator = self.fallback

    @classmethod
    def from_index(cls, settings: Settings | None = None) -> LeafletAssistant:
        settings = settings or get_settings()
        return cls(load_retriever(settings.index_dir), settings)

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        result = self._ask(question.strip())
        result.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return result

    def _ask(self, question: str) -> Answer:
        guard = check_question(question)
        if not guard.allowed:
            return Answer(question, guard.message or "", f"refused_{guard.category}")

        hits = self.retriever.search(question, k=self.settings.top_k)
        top = hits[0].score if hits else 0.0
        hits = _focus_on_medicine(question, [h for h in hits if h.score >= self.settings.min_score])
        if not hits:
            return Answer(question, NO_ANSWER, "not_found", top_score=round(top, 3))

        generator = self.generator
        try:
            text = generator.generate(question, hits)
        except Exception:  # network/API errors: degrade to extractive answer instead of failing
            logger.exception("LLM generation failed; using extractive fallback")
            generator = self.fallback
            text = generator.generate(question, hits)

        if text.strip() == NOT_FOUND:
            return Answer(question, NO_ANSWER, "not_found", generator=generator.name, top_score=round(top, 3))

        used = sorted(n for n in cited_numbers(text) if 1 <= n <= len(hits))
        citations = [_citation(n, hits[n - 1]) for n in used]
        return Answer(question, text, "answered", citations, generator.name, round(top, 3))


def _focus_on_medicine(question: str, hits: list[Hit]) -> list[Hit]:
    """Keep only passages from the medicine the question is about.

    Mixing leaflets is a safety risk (e.g. quoting another medicine's storage temperature or
    dose). If the question names one or more leaflets, keep those; otherwise keep the leaflet
    of the best-scoring passage.
    """
    if not hits:
        return hits
    q_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    named = {h.chunk.doc for h in hits if _medicine_name(h.chunk.doc) in q_words}
    keep = named or {hits[0].chunk.doc}
    return [h for h in hits if h.chunk.doc in keep]


def _medicine_name(doc: str) -> str:
    """First word of the leaflet file name, e.g. 'paracetamol-500mg-tablets' -> 'paracetamol'."""
    return re.split(r"[-_ ]", doc.lower())[0]


def _citation(number: int, hit: Hit) -> Citation:
    c = hit.chunk
    return Citation(number, c.doc, c.section, c.section_title, round(hit.score, 3), c.text)
