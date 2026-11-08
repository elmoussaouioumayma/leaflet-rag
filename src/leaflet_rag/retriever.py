"""Retrieval over leaflet chunks.

Two interchangeable backends:
* ``tfidf``      - scikit-learn TF-IDF with word and character n-grams. No model download,
                   fast, and a strong baseline for drug names and medical terms.
* ``embeddings`` - sentence-transformers dense embeddings (optional dependency).
Both return cosine similarities in [0, 1], so the same ``min_score`` threshold applies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    backend = "base"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def search(self, query: str, k: int = 4) -> list[Hit]:
        scores = self._scores(query)
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]

    def _scores(self, query: str) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- persistence -------------------------------------------------------
    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        (index_dir / "meta.json").write_text(json.dumps({"backend": self.backend}))
        self._save_state(index_dir)

    def _save_state(self, index_dir: Path) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class TfidfRetriever(Retriever):
    backend = "tfidf"

    def __init__(self, chunks: list[Chunk], fit: bool = True, word_weight: float = 0.7):
        super().__init__(chunks)
        self.word_weight = word_weight
        if fit:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize

            texts = [_index_text(c) for c in chunks]
            # Word n-grams carry meaning; character n-grams make drug names robust to
            # spelling and inflection. Scores are blended so off-topic questions, which only
            # share character fragments, stay below the "not found" threshold.
            self.word_vec = TfidfVectorizer(
                ngram_range=(1, 2), sublinear_tf=True, tokenizer=_stem_tokens, token_pattern=None
            )
            self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self.word_matrix = normalize(self.word_vec.fit_transform(texts))
            self.char_matrix = normalize(self.char_vec.fit_transform(texts))

    def _scores(self, query: str) -> np.ndarray:
        from sklearn.preprocessing import normalize

        word = (self.word_matrix @ normalize(self.word_vec.transform([query])).T).toarray().ravel()
        char = (self.char_matrix @ normalize(self.char_vec.transform([query])).T).toarray().ravel()
        return self.word_weight * word + (1 - self.word_weight) * char

    def _save_state(self, index_dir: Path) -> None:
        joblib.dump(
            {k: getattr(self, k) for k in ("word_vec", "char_vec", "word_matrix", "char_matrix", "word_weight")},
            index_dir / "tfidf.joblib",
        )


class EmbeddingRetriever(Retriever):
    backend = "embeddings"

    def __init__(self, chunks: list[Chunk], model_name: str, fit: bool = True):
        super().__init__(chunks)
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        if fit:
            self.matrix = self.model.encode(
                [_index_text(c) for c in chunks], normalize_embeddings=True, show_progress_bar=False
            )

    def _scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.clip(self.matrix @ q, 0.0, 1.0)

    def _save_state(self, index_dir: Path) -> None:
        np.save(index_dir / "embeddings.npy", self.matrix)
        (index_dir / "embedding_model.txt").write_text(self.model_name)


# Words patients use for each QRD section, added to the indexed text (field boosting), so
# "how should X be stored" finds section 5 even if the leaflet only says "Do not store above...".
SECTION_HINTS = {
    "1": "what is it used for indication treat",
    "2": "before taking warnings precautions interactions other medicines pregnancy breastfeeding driving alcohol",
    "3": "how to take dose dosage swallow tablets per day maximum forget missed",
    "4": "side effects adverse reactions common rare",
    "5": "storage store temperature keep",
    "6": "ingredients contains pack size appearance",
}

_SUFFIXES = ("ings", "ing", "ies", "ed", "es", "s", "e")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _stem_tokens(text: str) -> list[str]:
    """Lowercase word tokens without stop words, with light suffix stripping (capsule/capsules -> capsul)."""
    import re

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS]


def _index_text(chunk: Chunk) -> str:
    # Prefix the medicine name and section so "side effects of X" matches the right chunk.
    hints = SECTION_HINTS.get(chunk.section, "")
    return f"{chunk.doc.replace('-', ' ')} | {chunk.section_title} | {hints} | {chunk.text}"


def build_retriever(chunks: list[Chunk], backend: str, embedding_model: str) -> Retriever:
    if backend == "tfidf":
        return TfidfRetriever(chunks)
    if backend == "embeddings":
        return EmbeddingRetriever(chunks, embedding_model)
    raise ValueError(f"Unknown retriever backend: {backend}")


def load_retriever(index_dir: Path) -> Retriever:
    meta = json.loads((index_dir / "meta.json").read_text())
    chunks = [Chunk(**c) for c in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))]
    if meta["backend"] == "tfidf":
        retriever = TfidfRetriever(chunks, fit=False)
        for key, value in joblib.load(index_dir / "tfidf.joblib").items():
            setattr(retriever, key, value)
        return retriever
    if meta["backend"] == "embeddings":
        model_name = (index_dir / "embedding_model.txt").read_text().strip()
        retriever = EmbeddingRetriever(chunks, model_name, fit=False)
        retriever.matrix = np.load(index_dir / "embeddings.npy")
        return retriever
    raise ValueError(f"Unknown backend in index: {meta['backend']}")
