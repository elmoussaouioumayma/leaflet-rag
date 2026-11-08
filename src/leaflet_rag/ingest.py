"""Load package leaflets (PDF or text) and split them into section-aware chunks.

EU package leaflets follow the QRD template, with six numbered sections
("1. What X is and what it is used for", "2. What you need to know before ...",
"3. How to take ...", "4. Possible side effects", "5. How to store ...",
"6. Contents of the pack and other information"). Chunking along these
sections keeps each chunk on one topic and lets every answer cite its section.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

SECTION_RE = re.compile(
    r"^\s*([1-6])\.\s+(What .{3,80}|How to .{3,80}|Possible side effects.{0,40}|Contents of the pack.{0,60})\s*$",
    re.IGNORECASE,
)

SECTION_NAMES = {
    "1": "What it is and what it is used for",
    "2": "Before you take it",
    "3": "How to take it",
    "4": "Possible side effects",
    "5": "How to store it",
    "6": "Contents of the pack and other information",
}


@dataclass
class Chunk:
    chunk_id: str
    doc: str  # file stem, e.g. "paracetamol-500mg"
    section: str  # "1".."6" or "0" for text before the first section
    section_title: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def read_document(path: Path) -> str:
    """Return the plain text of a .pdf, .txt or .md file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported file type: {path.name}")


def _clean(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens from PDFs
    text = re.sub(r"-\n(?=[a-z])", "", text)  # words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


LEAFLET_START_RE = re.compile(r"package\s+leaflet\s*:\s*information\s+for\s+the\s+(user|patient)", re.IGNORECASE)


def leaflet_part(text: str) -> str:
    """Keep only the patient leaflet.

    Full EU product-information PDFs contain the SmPC (for professionals) and labelling before
    the package leaflet (Annex III B). The leaflet starts at the last
    "Package leaflet: Information for the user/patient" heading.
    """
    matches = list(LEAFLET_START_RE.finditer(text))
    return text[matches[-1].start() :] if matches else text


def split_sections(text: str) -> list[tuple[str, str, str]]:
    """Split leaflet text into (section_number, title, body) tuples."""
    lines = text.splitlines()
    # Leaflets list the six section titles in a table of contents before the real sections,
    # so a heading number can appear twice. The last occurrence of each number is the real one.
    last_seen: dict[str, int] = {}
    for idx, line in enumerate(lines):
        match = SECTION_RE.match(line)
        if match:
            last_seen[match.group(1)] = idx
    starts: dict[int, str] = {}
    previous = 0
    for idx, number in sorted((i, n) for n, i in last_seen.items()):
        if int(number) > previous:  # keep headings in increasing order only
            starts[idx] = number
            previous = int(number)

    sections: list[tuple[str, str, list[str]]] = [("0", "Introduction", [])]
    for idx, line in enumerate(lines):
        if idx in starts:
            sections.append((starts[idx], line.strip(), []))
        else:
            sections[-1][2].append(line)
    result = []
    for number, title, lines in sections:
        if number == "0":  # drop table-of-contents lines from the introduction
            lines = [line for line in lines if not SECTION_RE.match(line)]
        body = _clean("\n".join(lines))
        if body:
            result.append((number, title, body))
    return result


def _windows(text: str, max_words: int, overlap: int) -> list[str]:
    """Split long text into overlapping word windows, preferring paragraph breaks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], []
    for paragraph in paragraphs:
        words = paragraph.split()
        if current and len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(words)
        while len(current) > max_words:
            chunks.append(" ".join(current[:max_words]))
            current = current[max_words - overlap :]
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(doc: str, text: str, max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for number, title, body in split_sections(text):
        for i, window in enumerate(_windows(body, max_words, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc}:{number}:{i}",
                    doc=doc,
                    section=number,
                    section_title=SECTION_NAMES.get(number, title),
                    text=window,
                )
            )
    return chunks


def load_corpus(raw_dir: Path) -> list[Chunk]:
    files = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in {".pdf", ".txt", ".md"})
    if not files:
        raise FileNotFoundError(f"No .pdf/.txt/.md leaflets found in {raw_dir}")
    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(chunk_document(path.stem, leaflet_part(read_document(path))))
    return chunks
