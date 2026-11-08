"""Rule-based safety checks applied before retrieval.

The assistant only reports what package leaflets say. It must not:
* handle emergencies (overdose, poisoning, severe reactions, self-harm) - it points to emergency help;
* make personal treatment decisions ("should I take ...", "is it safe for me ...") - it points
  to a pharmacist or doctor.

Rules are deliberately simple and transparent so their behaviour can be tested and audited
(see eval/guardrail_questions.jsonl). They are a first line of defence, not a medical classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EMERGENCY_PATTERNS = [
    r"\boverdos\w*",
    r"\b(took|taken|swallowed|ate)\b.{0,30}\b(too many|too much|whole (pack|bottle|box)|all (the|my))\b",
    r"\bpoison\w*",
    r"\b(can'?t|cannot|unable to) breathe\b",
    r"\b(trouble|difficulty) breathing\b",
    r"\banaphyla\w*",
    r"\b(swollen|swelling of (the|my)) (face|throat|tongue|lips)\b",
    r"\bunconscious\b|\bpassed out\b|\bseizure\b",
    r"\b(kill myself|suicid\w*|end my life|self[- ]harm)\b",
]

PERSONAL_ADVICE_PATTERNS = [
    r"\bshould i\b",
    r"\bshall i\b",
    r"\bdo i need to\b",
    r"\bis it (safe|ok|okay|fine) (for me|for my|if i)\b",
    r"\b(can|may) i (take|use|give|combine|mix|double|stop)\b.{0,60}\b(my|i have|i'?m|i am)\b",
    r"\bhow (much|many)\b.{0,40}\b(should i|do i|for me|for my)\b",
    r"\bwhat should i (take|do|use)\b",
    r"\bdiagnos\w*\b",
    r"\bdo i have\b",
]

EMERGENCY_MESSAGE = (
    "This sounds like it could be an emergency. Please call your local emergency number "
    "(112 in the EU) or contact a poison control centre now. This assistant cannot help with emergencies."
)

PERSONAL_ADVICE_MESSAGE = (
    "I can only report what package leaflets say, not decide what is right for you personally. "
    "Please ask a pharmacist or your doctor. You can ask me, for example, what a leaflet says "
    "about side effects, interactions or storage."
)


@dataclass
class GuardrailResult:
    allowed: bool
    category: str  # "ok", "emergency", "personal_advice"
    message: str | None = None


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


# General questions that every leaflet answers in section 3, even though they contain "should I".
LEAFLET_FAQ_PATTERNS = [
    r"\b(forget|forgot|forgotten|miss|missed)\b.{0,20}\b(dose|tablet|capsule)\b",
]


def check_question(question: str) -> GuardrailResult:
    text = question.lower().strip()
    if _matches(EMERGENCY_PATTERNS, text):
        return GuardrailResult(False, "emergency", EMERGENCY_MESSAGE)
    if _matches(LEAFLET_FAQ_PATTERNS, text) and not re.search(r"\b(double|my \w+ (is|has))\b", text):
        return GuardrailResult(True, "ok")
    if _matches(PERSONAL_ADVICE_PATTERNS, text):
        return GuardrailResult(False, "personal_advice", PERSONAL_ADVICE_MESSAGE)
    return GuardrailResult(True, "ok")
