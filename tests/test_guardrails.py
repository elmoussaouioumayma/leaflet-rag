from pathlib import Path

import pytest

from leaflet_rag.evaluate import load_questions
from leaflet_rag.guardrails import check_question

GUARDRAIL_SET = load_questions(Path(__file__).parents[1] / "eval" / "guardrail_questions.jsonl")


@pytest.mark.parametrize("row", GUARDRAIL_SET, ids=[r["id"] for r in GUARDRAIL_SET])
def test_guardrail_set(row):
    assert check_question(row["question"]).category == row["type"]


@pytest.mark.parametrize(
    "question",
    [
        "What are the side effects of Fictamol?",
        "Can Testoprazole be taken during pregnancy?",
        "How should Sampleastine be stored?",
        "What should I do if I forget a dose of Testoprazole?",
        "Does Sampleastine interact with alcohol?",
    ],
)
def test_general_leaflet_questions_allowed(question):
    assert check_question(question).allowed
