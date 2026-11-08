from dataclasses import replace
from pathlib import Path

import pytest

from leaflet_rag.config import get_settings
from leaflet_rag.ingest import load_corpus
from leaflet_rag.pipeline import LeafletAssistant
from leaflet_rag.retriever import TfidfRetriever

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def chunks():
    return load_corpus(FIXTURES)


@pytest.fixture(scope="session")
def retriever(chunks):
    return TfidfRetriever(chunks)


@pytest.fixture()
def settings(tmp_path):
    # No LLM in tests: the extractive generator keeps tests offline and deterministic.
    return replace(get_settings(), raw_dir=FIXTURES, index_dir=tmp_path / "index", llm_api_key=None, llm_base_url=None)


@pytest.fixture()
def assistant(retriever, settings):
    return LeafletAssistant(retriever, settings)
