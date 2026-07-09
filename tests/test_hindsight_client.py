"""Tests for HindsightClient — mock httpx transports, no live server required."""

import asyncio
import pytest

import httpx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockTransport(httpx.MockTransport if hasattr(httpx, "MockTransport") else object):
    """Minimal sync/async transport that returns a canned response."""


def _sync_transport(status: int, body: bytes = b"{}"):
    """Return an httpx transport that always replies with (status, body)."""

    class _T(httpx.BaseTransport):
        def handle_request(self, request):
            return httpx.Response(status, content=body)

    return _T()


def _async_transport(status: int, body: bytes = b"{}"):
    """Return an async httpx transport that always replies with (status, body)."""

    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(status, content=body)

    return _T()


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _parse_recall
# ---------------------------------------------------------------------------

def test_parse_recall_list_of_dicts():
    from src.hindsight_client import _parse_recall

    data = [
        {"content": "Alice loves hiking", "score": 0.9},
        {"text": "Bob drinks espresso", "relevance": 0.7},
    ]
    results = _parse_recall(data, top_k=5)
    assert [r["text"] for r in results] == ["Alice loves hiking", "Bob drinks espresso"]
    assert results[0]["score"] == 0.9
    assert results[1]["score"] == 0.7


def test_parse_recall_wrapped_in_results_key():
    from src.hindsight_client import _parse_recall

    data = {"results": [{"content": "fact one"}, {"content": "fact two"}]}
    results = _parse_recall(data, top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "fact one"


def test_parse_recall_string_items():
    from src.hindsight_client import _parse_recall

    data = ["memory A", "memory B", ""]
    results = _parse_recall(data, top_k=10)
    assert [r["text"] for r in results] == ["memory A", "memory B"]
    assert all(r["score"] is None for r in results)


def test_parse_recall_unknown_shape_returns_empty():
    from src.hindsight_client import _parse_recall

    assert _parse_recall(None, top_k=5) == []
    assert _parse_recall("unexpected string", top_k=5) == []
    assert _parse_recall({}, top_k=5) == []


def test_parse_recall_skips_empty_content():
    from src.hindsight_client import _parse_recall

    data = [{"content": ""}, {"content": None}, {"content": "valid"}]
    results = _parse_recall(data, top_k=10)
    assert len(results) == 1
    assert results[0]["text"] == "valid"


# ---------------------------------------------------------------------------
# check_health
# ---------------------------------------------------------------------------

def test_check_health_marks_healthy_on_200():
    from src.hindsight_client import HindsightClient

    client = HindsightClient()
    client._get_async_client = lambda: httpx.AsyncClient(transport=_async_transport(200))
    assert run(client.check_health()) is True
    assert client.healthy is True


def test_check_health_marks_unhealthy_on_500():
    from src.hindsight_client import HindsightClient

    client = HindsightClient()
    client._get_async_client = lambda: httpx.AsyncClient(transport=_async_transport(500))
    assert run(client.check_health()) is False
    assert client.healthy is False


def test_check_health_marks_unhealthy_on_connection_error():
    from src.hindsight_client import HindsightClient

    class _ErrorTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("refused")

    client = HindsightClient()
    client._get_async_client = lambda: httpx.AsyncClient(transport=_ErrorTransport())
    assert run(client.check_health()) is False
    assert client.healthy is False


# ---------------------------------------------------------------------------
# retain
# ---------------------------------------------------------------------------

def test_retain_skipped_when_not_healthy():
    from src.hindsight_client import HindsightClient

    client = HindsightClient()
    # healthy stays False by default
    result = run(client.retain("test memory"))
    assert result is False


def test_retain_posts_correct_payload():
    from src.hindsight_client import HindsightClient

    captured = []

    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            import json as _json
            captured.append(_json.loads(request.content))
            return httpx.Response(200, content=b"{}")

    client = HindsightClient(bank_id="mybank")
    client.healthy = True
    client._get_async_client = lambda: httpx.AsyncClient(transport=_T())

    run(client.retain("User likes coffee", metadata={"category": "preference"}))

    assert len(captured) == 1
    assert captured[0]["bank_id"] == "mybank"
    assert captured[0]["content"] == "User likes coffee"
    assert captured[0]["metadata"]["category"] == "preference"


def test_retain_returns_false_on_server_error():
    from src.hindsight_client import HindsightClient

    client = HindsightClient()
    client.healthy = True
    client._get_async_client = lambda: httpx.AsyncClient(transport=_async_transport(500))
    assert run(client.retain("test")) is False


def test_retain_swallows_connection_error():
    from src.hindsight_client import HindsightClient

    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("refused")

    client = HindsightClient()
    client.healthy = True
    client._get_async_client = lambda: httpx.AsyncClient(transport=_T())
    assert run(client.retain("test")) is False  # no exception raised


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

def test_recall_skipped_when_not_healthy():
    from src.hindsight_client import HindsightClient

    client = HindsightClient()
    assert client.recall("anything") == []


def test_recall_returns_parsed_hits():
    import json as _json
    from src.hindsight_client import HindsightClient

    body = _json.dumps({"results": [{"content": "fact A", "score": 0.8}]}).encode()
    client = HindsightClient(bank_id="testbank")
    client.healthy = True
    client._sync_client = httpx.Client(transport=_sync_transport(200, body))

    hits = client.recall("query", top_k=5)
    assert len(hits) == 1
    assert hits[0]["text"] == "fact A"
    assert hits[0]["score"] == 0.8


def test_recall_returns_empty_on_server_error():
    from src.hindsight_client import HindsightClient

    client = HindsightClient()
    client.healthy = True
    client._sync_client = httpx.Client(transport=_sync_transport(500))
    assert client.recall("query") == []


def test_recall_swallows_connection_error():
    from src.hindsight_client import HindsightClient

    class _T(httpx.BaseTransport):
        def handle_request(self, request):
            raise httpx.ConnectError("refused")

    client = HindsightClient()
    client.healthy = True
    client._sync_client = httpx.Client(transport=_T())
    assert client.recall("query") == []  # no exception raised
