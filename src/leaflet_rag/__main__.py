"""Command line: python -m leaflet_rag {ingest,ask,eval,serve,ui}"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="leaflet_rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Read leaflets from LEAFLET_RAW_DIR and build the search index")

    ask_p = sub.add_parser("ask", help="Ask one question from the command line")
    ask_p.add_argument("question")

    eval_p = sub.add_parser("eval", help="Run the evaluation set and write a report")
    eval_p.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    eval_p.add_argument("--guardrails", type=Path, default=Path("eval/guardrail_questions.jsonl"))
    eval_p.add_argument("--out", type=Path, default=Path("eval/results"))
    eval_p.add_argument("--mlflow", action="store_true", help="Also log params, metrics and report to MLflow")
    eval_p.add_argument("--min-guardrail-accuracy", type=float, default=None, help="Exit 1 if below (for CI)")

    serve_p = sub.add_parser("serve", help="Run the FastAPI service")
    serve_p.add_argument("--port", type=int, default=8000)

    sub.add_parser("ui", help="Run the Gradio interface")

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "ingest":
        from .ingest import load_corpus
        from .retriever import build_retriever

        chunks = load_corpus(settings.raw_dir)
        retriever = build_retriever(chunks, settings.retriever, settings.embedding_model)
        retriever.save(settings.index_dir)
        docs = sorted({c.doc for c in chunks})
        print(f"Indexed {len(chunks)} chunks from {len(docs)} leaflets ({retriever.backend}) -> {settings.index_dir}")
        return 0

    if args.command == "ask":
        from .pipeline import LeafletAssistant

        answer = LeafletAssistant.from_index(settings).ask(args.question)
        print(json.dumps(asdict(answer), indent=2, ensure_ascii=False))
        return 0

    if args.command == "eval":
        from .evaluate import evaluate, load_questions, log_to_mlflow, write_report
        from .pipeline import LeafletAssistant

        questions = []
        for path in (args.questions, args.guardrails):
            if path and path.exists():
                questions += load_questions(path)
        if not questions:
            print("No evaluation questions found.", file=sys.stderr)
            return 1
        assistant = LeafletAssistant.from_index(settings)
        metrics, per_question = evaluate(assistant, questions)
        report = write_report(metrics, per_question, args.out)
        if args.mlflow:
            log_to_mlflow(metrics, settings, args.out)
        print(json.dumps(metrics, indent=2))
        print(f"Report: {report}")
        threshold = args.min_guardrail_accuracy
        if threshold is not None and (metrics["guardrail_accuracy"] or 0) < threshold:
            print(f"Guardrail accuracy below {threshold}", file=sys.stderr)
            return 1
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run("leaflet_rag.api:app", host="0.0.0.0", port=args.port)
        return 0

    if args.command == "ui":
        from .ui import main as ui_main

        ui_main()
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
