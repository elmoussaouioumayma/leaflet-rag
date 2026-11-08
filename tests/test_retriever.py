from leaflet_rag.retriever import TfidfRetriever, load_retriever


def test_finds_right_leaflet_and_section(retriever):
    hit = retriever.search("side effects of Fictamol", k=1)[0]
    assert hit.chunk.doc == "fictamol-250mg"
    assert hit.chunk.section == "4"


def test_scores_sorted_and_bounded(retriever):
    hits = retriever.search("how to store Testoprazole", k=4)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_off_topic_query_scores_low(retriever):
    assert retriever.search("who won the football world cup", k=1)[0].score < 0.12


def test_save_and_load_roundtrip(retriever, tmp_path):
    retriever.save(tmp_path)
    loaded = load_retriever(tmp_path)
    assert isinstance(loaded, TfidfRetriever)
    a = retriever.search("Sampleastine drowsy", k=3)
    b = loaded.search("Sampleastine drowsy", k=3)
    assert [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b]


def test_dotenv_loading(tmp_path, monkeypatch):
    from leaflet_rag.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text("# comment\nLEAFLET_TOP_K=7   # inline comment\nLLM_MODEL='m1'\n")
    monkeypatch.delenv("LEAFLET_TOP_K", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["LEAFLET_TOP_K"] == "7" and os.environ["LLM_MODEL"] == "m1"
