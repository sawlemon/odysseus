"""Thin HTTP client for the Hindsight local memory server.

retain() is async (can take 15-20s on Ollama) — callers must fire it via
asyncio.create_task() so it never blocks a request or chat turn.

recall() is intentionally sync to fit the synchronous build_context_preface
path in chat_processor.py. All exceptions are swallowed and logged at DEBUG
level so a downed Hindsight server never surfaces to end users.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class HindsightClient:
    """Wraps the Hindsight REST API (default: http://localhost:8888).

    Mirrors the EmbeddingClient pattern from src/embeddings.py: a `healthy`
    flag gates every call so the app degrades silently when Hindsight is
    absent, exactly as it does for ChromaDB.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8888",
        bank_id: str = "odysseus",
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.bank_id = bank_id
        self._headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self.healthy: bool = False
        self._warned: bool = False

        # Sync client: recall() on the synchronous context-preface path
        self._sync_client = httpx.Client(
            timeout=httpx.Timeout(connect=2.0, read=5.0, write=3.0, pool=2.0),
        )
        # Async client: retain() / ensure_bank() / reflect() as background tasks
        self._async_client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
            )
        return self._async_client

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def check_health(self) -> bool:
        """Probe the server and update self.healthy. Called once on startup."""
        try:
            client = self._get_async_client()
            resp = await client.get(f"{self.base_url}/")
            self.healthy = resp.status_code < 500
        except Exception:
            self.healthy = False
        if not self.healthy and not self._warned:
            logger.warning(
                "Hindsight unavailable at %s — memory mirroring disabled "
                "(set HINDSIGHT_BASE_URL / HINDSIGHT_ENABLED to reconfigure)",
                self.base_url,
            )
            self._warned = True
        return self.healthy

    async def ensure_bank(self) -> None:
        """Best-effort idempotent bank creation. Errors are silently ignored."""
        try:
            client = self._get_async_client()
            await client.post(
                f"{self.base_url}/banks",
                headers=self._headers,
                json={"bank_id": self.bank_id, "name": self.bank_id},
            )
        except Exception:
            pass  # bank probably already exists; ignore

    def close(self) -> None:
        """Release HTTP clients on app shutdown."""
        try:
            self._sync_client.close()
        except Exception:
            pass
        if self._async_client and not self._async_client.is_closed:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._async_client.aclose())
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def retain(
        self,
        text: str,
        *,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """Store a memory in Hindsight. Always fire via asyncio.create_task()."""
        if not self.healthy:
            return False
        payload: Dict[str, Any] = {"bank_id": self.bank_id, "content": text}
        if context:
            payload["context"] = context
        if metadata:
            payload["metadata"] = metadata
        if timestamp:
            payload["timestamp"] = timestamp
        try:
            client = self._get_async_client()
            resp = await client.post(
                f"{self.base_url}/retain",
                headers=self._headers,
                json=payload,
            )
            return resp.is_success
        except Exception as exc:
            logger.debug("Hindsight retain failed: %s", exc)
            return False

    def recall(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search Hindsight for relevant memories. Sync; safe on the preface path."""
        if not self.healthy:
            return []
        try:
            resp = self._sync_client.post(
                f"{self.base_url}/recall",
                headers=self._headers,
                json={"bank_id": self.bank_id, "query": query, "budget": "low"},
            )
            if not resp.is_success:
                return []
            return _parse_recall(resp.json(), top_k)
        except Exception as exc:
            logger.debug("Hindsight recall failed: %s", exc)
            return []

    async def reflect(self, query: str) -> Optional[str]:
        """Synthesize memories via Hindsight reflect. Async; for future tool use."""
        if not self.healthy:
            return None
        try:
            client = self._get_async_client()
            resp = await client.post(
                f"{self.base_url}/reflect",
                headers=self._headers,
                json={"bank_id": self.bank_id, "query": query},
            )
            if not resp.is_success:
                return None
            data = resp.json()
            if isinstance(data, dict):
                return (
                    data.get("content")
                    or data.get("result")
                    or data.get("response")
                    or data.get("synthesis")
                )
            return str(data) if data else None
        except Exception as exc:
            logger.debug("Hindsight reflect failed: %s", exc)
            return None


def _parse_recall(data: Any, top_k: int) -> List[Dict[str, Any]]:
    """Parse recall response defensively — accept a few plausible shapes."""
    items: Optional[List[Any]] = None

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("results", "memories", "items", "data"):
            if isinstance(data.get(key), list):
                items = data[key]
                break

    if items is None:
        return []

    results: List[Dict[str, Any]] = []
    for item in items[:top_k]:
        if isinstance(item, dict):
            text = (
                item.get("content")
                or item.get("text")
                or item.get("memory")
                or ""
            )
            if not text:
                continue
            score = (
                item.get("score")
                or item.get("relevance")
                or item.get("similarity")
            )
            results.append({"text": str(text), "score": score})
        elif isinstance(item, str) and item:
            results.append({"text": item, "score": None})

    return results
