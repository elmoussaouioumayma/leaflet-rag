"""Offline evaluation of retrieval, answers, abstention and guardrails.

Input: a JSONL file, one question per line:
    {"id": "q01", "type": "answerable", "question": "...",
     "expected_doc": "paracetamol-500mg", "expected_section": "4",
     "answer_keywords": ["nausea", "rash"]}
``type`` is one of: answerable, unanswerable, emergency, personal_advice.

Output: a metrics dict, written as JSON and as a Markdown report; optionally logged to MLflow.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path

from .pipeline import LeafletAssistant

EXPECTED_STATUS = {
    "unanswerable": "not_found",
    "emergency": "refused_emergency",
    "personal_advice": "refused_personal_advice",
}


def load_questions(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            rows.append(json.loads(line))
    return rows


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def evaluate(assistant: LeafletAssistant, questions: list[dict], k: int | None = None) -> tuple[dict, list[dict]]:
    k = k or assistant.settings.top_k
    per_question: list[dict] = []
    retr_doc_at1, retr_doc_atk, retr_sec_atk, rr = [], [], [], []
    kw_recall, answered, cited, cite_correct = [], [], [], []
    abstain_ok, guard_ok, false_refusal, latencies = [], [], [], []

    for q in questions:
        qtype = q.get("type", "answerable")
        result = assistant.ask(q["question"])
        latencies.append(result.latency_ms)
        row = {"id": q.get("id"), "type": qtype, "status": result.status, "answer": result.answer}

        if qtype == "answerable":
            hits = assistant.retriever.search(q["question"], k=k)
            docs = [h.chunk.doc for h in hits]
            pairs = [(h.chunk.doc, h.chunk.section) for h in hits]
            exp_doc, exp_sec = q["expected_doc"], str(q.get("expected_section", ""))
            retr_doc_at1.append(bool(docs) and docs[0] == exp_doc)
            retr_doc_atk.append(exp_doc in docs)
            if exp_sec:
                retr_sec_atk.append((exp_doc, exp_sec) in pairs)
            rank = next((i for i, d in enumerate(docs, start=1) if d == exp_doc), None)
            rr.append(1 / rank if rank else 0.0)

            is_answered = result.status == "answered"
            answered.append(is_answered)
            false_refusal.append(result.status.startswith("refused"))
            keywords = [w.lower() for w in q.get("answer_keywords", [])]
            if keywords:
                found = sum(w in result.answer.lower() for w in keywords)
                kw_recall.append(found / len(keywords))
                row["keyword_recall"] = round(found / len(keywords), 3)
            if is_answered:
                cited.append(bool(result.citations))
                if result.citations:
                    cite_correct.append(all(c.doc == exp_doc for c in result.citations))
            row["retrieved_docs"] = docs
        else:
            ok = result.status == EXPECTED_STATUS[qtype]
            (abstain_ok if qtype == "unanswerable" else guard_ok).append(ok)
            row["correct"] = ok
        per_question.append(row)

    n_by_type: dict[str, int] = {}
    for q in questions:
        n_by_type[q.get("type", "answerable")] = n_by_type.get(q.get("type", "answerable"), 0) + 1

    metrics = {
        "n_questions": len(questions),
        "n_by_type": n_by_type,
        "retriever": assistant.retriever.backend,
        "generator": assistant.generator.name,
        "k": k,
        "retrieval_doc_hit@1": _rate(retr_doc_at1),
        f"retrieval_doc_hit@{k}": _rate(retr_doc_atk),
        f"retrieval_section_hit@{k}": _rate(retr_sec_atk),
        "retrieval_mrr": round(statistics.mean(rr), 3) if rr else None,
        "answer_rate": _rate(answered),
        "answer_keyword_recall": round(statistics.mean(kw_recall), 3) if kw_recall else None,
        "citation_rate": _rate(cited),
        "citation_doc_precision": _rate(cite_correct),
        "abstention_accuracy": _rate(abstain_ok),
        "guardrail_accuracy": _rate(guard_ok),
        "false_refusal_rate": _rate(false_refusal),
        "latency_ms_p50": round(statistics.median(latencies), 1) if latencies else None,
        "latency_ms_p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1) if latencies else None,
    }
    return metrics, per_question


def write_report(metrics: dict, per_question: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "per_question.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in per_question), encoding="utf-8"
    )
    lines = ["# Evaluation report", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in metrics.items() if not isinstance(v, dict)]
    lines += ["", f"Questions by type: {metrics['n_by_type']}", "", "## Failures", ""]
    failures = [
        r
        for r in per_question
        if (r["type"] == "answerable" and (r["status"] != "answered" or r.get("keyword_recall", 1) < 0.5))
        or r.get("correct") is False
    ]
    lines += [f"- `{r['id']}` ({r['type']}): status={r['status']}" for r in failures] or ["None."]
    report = out_dir / "report.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def log_to_mlflow(metrics: dict, settings, report_dir: Path) -> None:
    import mlflow

    mlflow.set_experiment("leaflet-rag")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "retriever": metrics["retriever"],
                "generator": metrics["generator"],
                "k": metrics["k"],
                "min_score": settings.min_score,
                "llm_model": settings.llm_model if metrics["generator"] == "llm" else "none",
            }
        )
        mlflow.log_metrics(
            {k.replace("@", "_at_"): v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
        )
        mlflow.log_artifacts(str(report_dir))


def run(assistant: LeafletAssistant, questions_path: Path, out_dir: Path, use_mlflow: bool = False) -> dict:
    metrics, per_question = evaluate(assistant, load_questions(questions_path))
    write_report(metrics, per_question, out_dir)
    if use_mlflow:
        log_to_mlflow(metrics, assistant.settings, out_dir)
    return metrics


__all__ = ["evaluate", "load_questions", "run", "write_report", "asdict"]
