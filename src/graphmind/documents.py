"""Corpus validation and content addressing. All lengths count Unicode code points."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class GraphMindError(ValueError):
    def __init__(self, field: str, detail: str):
        self.field = field
        self.detail = detail
        super().__init__(f"{field}: {detail}")


def _string(value: object, field: str, *, nonblank: bool = False) -> str:
    if not isinstance(value, str) or (nonblank and not value.strip()):
        raise GraphMindError(
            field, "expected a nonblank string" if nonblank else "expected a string"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise GraphMindError(field, "expected Unicode scalar values") from exc
    return value


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    text: str

    def __post_init__(self) -> None:
        _string(self.id, "id", nonblank=True)
        _string(self.title, "title", nonblank=True)
        _string(self.text, "text")
        if len(self.text) > 50_000:
            raise GraphMindError("text", "exceeds 50,000 code points")


type Documents = Sequence[Document | Mapping[str, object]]


def digest(parts: Sequence[str | int]) -> str:
    serialized = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def document_id(title: str, text: str) -> str:
    return digest([_string(title, "title", nonblank=True), _string(text, "text")])


def validate_documents(documents: Documents) -> list[Document]:
    if (
        not isinstance(documents, Sequence)
        or isinstance(documents, (str, bytes))
        or not 1 <= len(documents) <= 100
    ):
        raise GraphMindError("documents", "expected an array of 1–100 paragraphs")
    seen: set[str] = set()
    total = 0
    result = []
    for row in documents:
        if isinstance(row, Document):
            doc = row
        elif isinstance(row, Mapping) and set(row) == {"id", "title", "text"}:
            doc = Document(
                _string(row["id"], "id"),
                _string(row["title"], "title"),
                _string(row["text"], "text"),
            )
        else:
            raise GraphMindError("documents", "each row must contain exactly id, title, text")
        if doc.id in seen:
            raise GraphMindError("id", f"duplicate paragraph ID {doc.id!r}")
        seen.add(doc.id)
        total += len(doc.text)
        if total > 1_000_000:
            raise GraphMindError("documents", "exceeds 1,000,000 total body code points")
        result.append(doc)
    return sorted(result, key=lambda doc: doc.id)


def validate_question(question: object) -> str:
    text = _string(question, "question", nonblank=True)
    if len(text) > 2_000:
        raise GraphMindError("question", "exceeds 2,000 code points")
    return text


STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "more",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "or",
        "our",
        "out",
        "over",
        "she",
        "should",
        "so",
        "some",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "under",
        "up",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)
