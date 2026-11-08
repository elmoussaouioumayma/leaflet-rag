from pathlib import Path

from leaflet_rag.evaluate import evaluate, load_questions, write_report
from leaflet_rag.generator import NOT_FOUND
from leaflet_rag.pipeline import LeafletAssistant

FIXTURE_EVAL = Path(__file__).parent / "fixtures" / "eval_fixture.jsonl"


def test_answer_has_citation_to_right_leaflet(assistant):
    result = assistant.ask("What are the side effects of Fictamol?")
    assert result.status == "answered"
    assert result.citations and result.citations[0].doc == "fictamol-250mg"
    assert "[1]" in result.answer or "[2]" in result.answer


def test_off_topic_returns_not_found(assistant):
    assert assistant.ask("What is the best recipe for chocolate cake?").status == "not_found"


def test_emergency_is_refused_before_retrieval(assistant):
    result = assistant.ask("I took too many Fictamol tablets")
    assert result.status == "refused_emergency"
    assert "112" in result.answer


class _FailingLLM:
    name = "llm"

    def generate(self, question, hits):
        raise ConnectionError("API down")


class _AbstainingLLM:
    name = "llm"

    def generate(self, question, hits):
        return NOT_FOUND


def test_llm_failure_falls_back_to_extractive(retriever, settings):
    result = LeafletAssistant(retriever, settings, generator=_FailingLLM()).ask("How to store Testoprazole?")
    assert result.status == "answered" and result.generator == "extractive"


def test_llm_abstention_maps_to_not_found(retriever, settings):
    result = LeafletAssistant(retriever, settings, generator=_AbstainingLLM()).ask("How to store Testoprazole?")
    assert result.status == "not_found"


def test_evaluation_on_fixture_set(assistant, tmp_path):
    questions = load_questions(FIXTURE_EVAL)
    metrics, per_question = evaluate(assistant, questions)
    assert metrics["retrieval_doc_hit@4"] == 1.0
    assert metrics["retrieval_section_hit@4"] == 1.0
    assert metrics["abstention_accuracy"] == 1.0
    assert metrics["citation_rate"] == 1.0
    report = write_report(metrics, per_question, tmp_path)
    assert report.exists() and (tmp_path / "metrics.json").exists()


def test_answer_never_mixes_medicines(assistant):
    result = assistant.ask("How should Testoprazole be stored?")
    assert {c.doc for c in result.citations} == {"testoprazole-20mg"}
    assert "25 degrees" not in result.answer  # storage temperature of the other fictional medicines


def test_llm_generator_prompt_and_parsing(retriever, settings, monkeypatch):
    from leaflet_rag.generator import LLMGenerator

    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("M", (), {"content": "Store it below 30 degrees Celsius [1]."})
            return type("R", (), {"choices": [type("C", (), {"message": message})]})

    generator = LLMGenerator("test-model", base_url="http://localhost:1/v1", api_key="x")
    generator.client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})})
    result = LeafletAssistant(retriever, settings, generator=generator).ask("How should Testoprazole be stored?")
    assert result.status == "answered" and result.generator == "llm"
    assert result.citations[0].doc == "testoprazole-20mg"
    prompt = captured["messages"][1]["content"]
    assert "[1] (testoprazole-20mg, section 5" in prompt
    assert captured["temperature"] == 0
