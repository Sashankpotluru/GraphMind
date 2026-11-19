import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from graphmind import Document, build_graph
from graphmind.documents import document_id


def test_content_id_and_frozen_source():
    payload = json.dumps(["É", "🚤"], ensure_ascii=False, separators=(",", ":")).encode()
    assert document_id("É", "🚤") == hashlib.sha256(payload).hexdigest()
    doc = Document("p", "É", "🚤")
    with pytest.raises(FrozenInstanceError):
        doc.text = "changed"


@pytest.mark.parametrize(
    "docs",
    [
        [],
        [Document("p", "T", "x")] * 2,
        [{"id": "x", "title": "X", "text": "x", "is_supporting": True}],
        [{"id": 1, "title": "X", "text": "x"}],
        [{"id": " ", "title": "X", "text": "x"}],
        [{"id": "x", "title": " ", "text": "text"}],
        [Document(str(i), "X", "x") for i in range(101)],
        [{"id": "p", "title": "X", "text": "x" * 50_001}],
        [Document(str(i), "X", "x" * 50_000) for i in range(21)],
    ],
)
def test_invalid_corpora(docs):
    with pytest.raises(ValueError):
        build_graph(docs)


def test_exact_size_limit_is_inclusive():
    assert len(build_graph([Document("p", "T", "x" * 50_000)])["paragraphs"]) == 1
