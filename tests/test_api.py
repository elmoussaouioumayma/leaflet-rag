from fastapi.testclient import TestClient

from leaflet_rag import api


def _client(assistant):
    client = TestClient(api.app)
    api._state["assistant"] = assistant  # inject the fixture index instead of loading from disk
    return client


def test_health(assistant):
    body = _client(assistant).get("/health").json()
    assert body["status"] == "ok" and body["chunks"] > 0


def test_ask_returns_answer_with_citations(assistant):
    response = _client(assistant).post("/ask", json={"question": "What is Sampleastine used for?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["citations"][0]["doc"] == "sampleastine-10mg"
    assert "not medical advice" in body["disclaimer"]


def test_question_validation(assistant):
    assert _client(assistant).post("/ask", json={"question": "x"}).status_code == 422


def test_metrics_exposed(assistant):
    client = _client(assistant)
    client.post("/ask", json={"question": "Should I take Fictamol for my headache?"})
    text = client.get("/metrics").text
    assert 'leaflet_requests_total{status="refused_personal_advice"}' in text
    assert "leaflet_latency_seconds" in text


def test_logs_do_not_contain_question_text(assistant, caplog):
    secret = "Fictamol for my rare condition XYZ123"
    with caplog.at_level("INFO", logger="leaflet_rag.api"):
        _client(assistant).post("/ask", json={"question": f"What are side effects of {secret}?"})
    assert "XYZ123" not in caplog.text
