"""Tests for format_docs (no embeddings / model required)."""

from dataclasses import dataclass, field

from rag import format_docs


@dataclass
class _Doc:
    page_content: str
    metadata: dict = field(default_factory=dict)


def test_format_docs_joins_and_labels():
    docs = [
        _Doc("hello", {"source": "A"}),
        _Doc("world", {"source": "B"}),
    ]
    out = format_docs(docs)
    assert "hello" in out
    assert "world" in out
    assert "[A]" in out
    assert "[B]" in out


def test_format_docs_handles_missing_source():
    out = format_docs([_Doc("x")])
    assert "x" in out
