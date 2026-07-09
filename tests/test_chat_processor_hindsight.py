"""Tests for Hindsight recall merge in ChatProcessor.build_context_preface."""

import asyncio
import pytest
from unittest.mock import MagicMock


class FakeHindsightClient:
    """Minimal in-process Hindsight client for ChatProcessor tests."""

    def __init__(self, healthy=True, recall_results=None):
        self.healthy = healthy
        self._recall_results = recall_results or []

    def recall(self, query, *, top_k=5):
        return self._recall_results[:top_k]

    async def retain(self, text, **kwargs):
        return True

    def close(self):
        pass


class FakeMemoryManager:
    def __init__(self, memories=None):
        self._memories = memories or []

    def load(self, owner=None):
        return self._memories

    def get_relevant_memories(self, query, memories, threshold=0.05, max_items=8):
        return memories[:max_items]

    def increment_uses(self, ids):
        pass


class FakePersonalDocsManager:
    rag_manager = None


# ---------------------------------------------------------------------------
# Hindsight recall merged when native finds nothing
# ---------------------------------------------------------------------------

def test_hindsight_recall_adds_context_when_native_empty(tmp_path):
    from src.chat_processor import ChatProcessor

    hindsight = FakeHindsightClient(
        recall_results=[{"text": "User is a backend engineer", "score": 0.9}]
    )
    mm = FakeMemoryManager(memories=[])  # no native memories
    cp = ChatProcessor(mm, FakePersonalDocsManager(), hindsight=hindsight)

    preface, _, _ = cp.build_context_preface(
        "what's my job?", session=None, use_memory=True, use_rag=False
    )

    texts = " ".join(m.get("content", "") for m in preface if isinstance(m, dict))
    assert "User is a backend engineer" in texts


def test_hindsight_recall_deduplicates_native_results(tmp_path):
    from src.chat_processor import ChatProcessor

    shared_text = "User prefers dark mode"
    mm = FakeMemoryManager(memories=[
        {"id": "n1", "text": shared_text, "category": "preference"},
    ])
    hindsight = FakeHindsightClient(
        recall_results=[{"text": shared_text, "score": 0.99}]
    )
    cp = ChatProcessor(mm, FakePersonalDocsManager(), hindsight=hindsight)

    preface, _, _ = cp.build_context_preface(
        "theme preference?", session=None, use_memory=True, use_rag=False
    )

    # shared_text should appear exactly once
    all_content = " | ".join(m.get("content", "") for m in preface if isinstance(m, dict))
    assert all_content.count(shared_text) == 1


def test_hindsight_recall_appends_new_results_after_native():
    from src.chat_processor import ChatProcessor

    mm = FakeMemoryManager(memories=[
        {"id": "n1", "text": "User works at Acme", "category": "fact"},
    ])
    hindsight = FakeHindsightClient(
        recall_results=[{"text": "User speaks French", "score": 0.85}]
    )
    cp = ChatProcessor(mm, FakePersonalDocsManager(), hindsight=hindsight)

    preface, _, _ = cp.build_context_preface(
        "languages?", session=None, use_memory=True, use_rag=False
    )

    # Hindsight result must appear; native BM25 may or may not retrieve for "languages?"
    all_content = " ".join(m.get("content", "") for m in preface if isinstance(m, dict))
    assert "User speaks French" in all_content


def test_hindsight_recall_skipped_when_use_memory_false():
    from src.chat_processor import ChatProcessor

    hindsight = FakeHindsightClient(
        recall_results=[{"text": "secret fact", "score": 1.0}]
    )
    cp = ChatProcessor(FakeMemoryManager(), FakePersonalDocsManager(), hindsight=hindsight)

    preface, _, _ = cp.build_context_preface(
        "anything", session=None, use_memory=False, use_rag=False
    )

    all_content = " ".join(m.get("content", "") for m in preface if isinstance(m, dict))
    assert "secret fact" not in all_content


def test_no_hindsight_does_not_break_preface():
    from src.chat_processor import ChatProcessor

    mm = FakeMemoryManager(memories=[
        {"id": "n1", "text": "User likes cats", "category": "preference"},
    ])
    # No hindsight (default None)
    cp = ChatProcessor(mm, FakePersonalDocsManager())

    preface, _, _ = cp.build_context_preface(
        "pets?", session=None, use_memory=True, use_rag=False
    )

    # Must return a list without raising and include at least the policy message
    assert isinstance(preface, list)
    assert len(preface) >= 1
    # No Hindsight content injected
    all_content = " ".join(m.get("content", "") for m in preface if isinstance(m, dict))
    assert "hindsight" not in all_content.lower()


def test_hindsight_recall_exception_does_not_raise():
    from src.chat_processor import ChatProcessor

    class BrokenHindsight:
        healthy = True

        def recall(self, query, *, top_k=5):
            raise RuntimeError("unexpected error")

    cp = ChatProcessor(FakeMemoryManager(), FakePersonalDocsManager(), hindsight=BrokenHindsight())

    # Should not raise, Hindsight errors are swallowed
    preface, _, _ = cp.build_context_preface(
        "test", session=None, use_memory=True, use_rag=False
    )
    assert isinstance(preface, list)
