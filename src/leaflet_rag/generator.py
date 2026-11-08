"""Answer generation from retrieved leaflet passages.

* ``LLMGenerator``        - any OpenAI-compatible chat API (OpenAI, Azure OpenAI, or a local
                            model through Ollama/vLLM via LLM_BASE_URL). The prompt restricts the
                            model to the given sources and requires [n] citations.
* ``ExtractiveGenerator`` - no LLM: returns the most relevant sentences verbatim with citations.
                            Used offline, in CI, and as a fallback if the LLM call fails.
"""

from __future__ import annotations

import re

from .retriever import Hit

SYSTEM_PROMPT = """You answer questions about medicines using ONLY the package-leaflet excerpts provided.
Rules:
- Use only facts stated in the excerpts. Never add outside medical knowledge.
- Cite every statement with the excerpt number in square brackets, e.g. [1] or [2][3].
- If the excerpts do not contain the answer, reply exactly: NOT_IN_SOURCES
- Do not give personal medical advice; report what the leaflet says.
- Be concise: at most 5 sentences, plain language."""

NOT_FOUND = "NOT_IN_SOURCES"


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.chunk.doc}, section {h.chunk.section}: {h.chunk.section_title})\n{h.chunk.text}"
        for i, h in enumerate(hits, start=1)
    )


def cited_numbers(answer: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[Hit]) -> str:
        from .retriever import SECTION_HINTS, _stem_tokens

        # The medicine name appears everywhere in its leaflet, so it says nothing about which
        # sentence answers the question; match on the remaining (stemmed) question words.
        name_terms = {t for h in hits for t in _stem_tokens(h.chunk.doc.replace("-", " "))}
        q_terms = set(_stem_tokens(question)) - name_terms
        scored = []
        for i, hit in enumerate(hits, start=1):
            if hit.chunk.section == "0":  # leaflet title page, rarely an answer
                continue
            hint_terms = set(_stem_tokens(SECTION_HINTS.get(hit.chunk.section, "")))
            section_bonus = 0.5 if q_terms & hint_terms else 0.0
            for position, sentence in enumerate(re.split(r"(?<=[.!?])\s+", hit.chunk.text)):
                overlap = len(q_terms & set(_stem_tokens(sentence)))
                # A question about a section's topic ("what is X used for") may share no words
                # with the answer sentence; then the opening sentences of that section answer it.
                if overlap or section_bonus:
                    lead = 0.1 / (position + 1)
                    scored.append((overlap + 2 * hit.score + section_bonus + lead, i, sentence.strip()))
        if not scored:
            return NOT_FOUND
        scored.sort(key=lambda x: -x[0])
        best = sorted(scored[: self.max_sentences], key=lambda x: x[1])  # keep source order
        return " ".join(f"{s} [{i}]" for _, i, s in best)


class LLMGenerator:
    name = "llm"

    def __init__(self, model: str, base_url: str | None, api_key: str | None, temperature: float = 0.0):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed-for-local-models")

    def generate(self, question: str, hits: list[Hit]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Excerpts:\n{format_sources(hits)}\n\nQuestion: {question}"},
            ],
        )
        return (response.choices[0].message.content or "").strip()
