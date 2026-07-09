"""Tests for HindsightMemoryProvider — MemoryRecord mapping, no-op delete/list."""

import asyncio
import pytest


def run(coro):
    return asyncio.run(coro)


class FakeHindsightClient:
    """Minimal in-process stand-in for HindsightClient."""

    bank_id = "test"
    base_url = "http://localhost:8888"

    def __init__(self, healthy=True):
        self.healthy = healthy
        self.retained = []
        self.recalled = []

    async def check_health(self):
        return self.healthy

    async def ensure_bank(self):
        pass

    async def retain(self, text, *, context=None, metadata=None, timestamp=None):
        if self.healthy:
            self.retained.append({"text": text, "metadata": metadata})
        return self.healthy

    def recall(self, query, *, top_k=5):
        return self.recalled[:top_k]

    def close(self):
        pass


# ---------------------------------------------------------------------------
# remember()
# ---------------------------------------------------------------------------

def test_remember_calls_retain_with_metadata():
    from src.hindsight_provider import HindsightMemoryProvider

    fake = FakeHindsightClient()
    provider = HindsightMemoryProvider(fake)

    record = run(provider.remember(
        "Alice dislikes loud music",
        owner="alice",
        session_id="s-1",
        category="preference",
        source="user",
        metadata={"confidence": 0.95},
    ))

    assert len(fake.retained) == 1
    call = fake.retained[0]
    assert call["text"] == "Alice dislikes loud music"
    meta = call["metadata"]
    assert meta["category"] == "preference"
    assert meta["source"] == "user"
    assert meta["owner"] == "alice"
    assert meta["session_id"] == "s-1"
    assert meta["confidence"] == 0.95


def test_remember_returns_memory_record():
    from src.hindsight_provider import HindsightMemoryProvider
    from src.memory_provider import MemoryRecord

    fake = FakeHindsightClient()
    provider = HindsightMemoryProvider(fake)
    record = run(provider.remember("some fact", category="fact"))

    assert isinstance(record, MemoryRecord)
    assert record.text == "some fact"
    assert record.category == "fact"


def test_remember_still_returns_record_when_client_unhealthy():
    from src.hindsight_provider import HindsightMemoryProvider
    from src.memory_provider import MemoryRecord

    fake = FakeHindsightClient(healthy=False)
    provider = HindsightMemoryProvider(fake)
    record = run(provider.remember("fact", category="identity"))

    assert isinstance(record, MemoryRecord)
    assert fake.retained == []  # nothing sent to unhealthy client


# ---------------------------------------------------------------------------
# recall()
# ---------------------------------------------------------------------------

def test_recall_maps_hits_to_memory_search_hits():
    from src.hindsight_provider import HindsightMemoryProvider
    from src.memory_provider import MemorySearchHit

    fake = FakeHindsightClient()
    fake.recalled = [
        {"text": "Alice works at Presidio", "score": 0.88},
        {"text": "Alice likes Python", "score": 0.75},
    ]
    provider = HindsightMemoryProvider(fake)

    hits = run(provider.recall("Alice", top_k=5))

    assert len(hits) == 2
    assert all(isinstance(h, MemorySearchHit) for h in hits)
    assert hits[0].provider_id == "hindsight"
    assert hits[0].memory.text == "Alice works at Presidio"
    assert hits[0].score == 0.88


def test_recall_returns_empty_when_unhealthy():
    from src.hindsight_provider import HindsightMemoryProvider

    fake = FakeHindsightClient(healthy=False)
    provider = HindsightMemoryProvider(fake)
    hits = run(provider.recall("anything"))
    assert hits == []


# ---------------------------------------------------------------------------
# list_memories() and delete() are intentional no-ops
# ---------------------------------------------------------------------------

def test_list_memories_returns_empty():
    from src.hindsight_provider import HindsightMemoryProvider

    provider = HindsightMemoryProvider(FakeHindsightClient())
    assert run(provider.list_memories()) == []


def test_delete_returns_false():
    from src.hindsight_provider import HindsightMemoryProvider

    provider = HindsightMemoryProvider(FakeHindsightClient())
    assert run(provider.delete("some-id")) is False


# ---------------------------------------------------------------------------
# enabled property
# ---------------------------------------------------------------------------

def test_enabled_reflects_client_health():
    from src.hindsight_provider import HindsightMemoryProvider

    healthy_client = FakeHindsightClient(healthy=True)
    sick_client = FakeHindsightClient(healthy=False)

    assert HindsightMemoryProvider(healthy_client).enabled is True
    assert HindsightMemoryProvider(sick_client).enabled is False


# ---------------------------------------------------------------------------
# initialize() / shutdown()
# ---------------------------------------------------------------------------

def test_initialize_runs_health_check_and_bank():
    from src.hindsight_provider import HindsightMemoryProvider

    fake = FakeHindsightClient(healthy=True)
    provider = HindsightMemoryProvider(fake)
    run(provider.initialize())  # should not raise


def test_shutdown_calls_close():
    from src.hindsight_provider import HindsightMemoryProvider

    class TrackingClient(FakeHindsightClient):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    client = TrackingClient()
    provider = HindsightMemoryProvider(client)
    run(provider.shutdown())
    assert client.closed is True


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_provider_registers_in_registry():
    from src.memory_provider import MemoryProviderRegistry
    from src.hindsight_provider import HindsightMemoryProvider

    fake = FakeHindsightClient(healthy=True)
    provider = HindsightMemoryProvider(fake)
    registry = MemoryProviderRegistry([provider])

    assert registry.get("hindsight") is provider
    # enabled reflects health
    assert provider in registry.active()
