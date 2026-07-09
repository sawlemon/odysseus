"""Thin HTTP client for the Hindsight local memory server.

retain() is async (can take 15-20s on Ollama) — callers must fire it via
asyncio.create_task() so it never blocks a request or chat turn.

recall() is intentionally sync to fit the synchronous build_context_preface
path in chat_processor.py. All exceptions are swallowed and logged at DEBUG
level so a downed Hindsight server never surfaces to end users.

Bank selection: both retain() and recall() ask the configured utility model
to pick the most appropriate bank(s) before writing/reading. Falls back
gracefully (self.bank_id for retain, all banks for recall) if the LLM call
fails or no utility model is configured.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_BANK_SELECT_WRITE_PROMPT = (
    "You are a memory bank router. "
    "Choose the single most appropriate bank for the memory below. "
    "Reply with only the bank_id — no punctuation, no explanation."
)
_BANK_SELECT_READ_PROMPT = (
    "You are a memory bank router. "
    "Choose the bank_id(s) most likely to contain relevant memories for the query below. "
    "Reply with a comma-separated list of bank_ids only — no explanation."
)


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

        self._bank_ids_cache: List[str] = []
        self._bank_ids_cached_at: float = 0.0

        # Sync client: recall() / _select_banks_for_read() on the synchronous path
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

    def _get_all_bank_ids(self) -> List[str]:
        """Return all bank IDs, refreshed from the server at most every 60s."""
        now = time.monotonic()
        if self._bank_ids_cache and (now - self._bank_ids_cached_at) < 60:
            return self._bank_ids_cache
        try:
            resp = self._sync_client.get(
                f"{self.base_url}/v1/default/banks",
                headers=self._headers,
            )
            if resp.is_success:
                banks = resp.json().get("banks", [])
                ids = [b["bank_id"] for b in banks if "bank_id" in b]
                if ids:
                    self._bank_ids_cache = ids
                    self._bank_ids_cached_at = now
                    return ids
        except Exception as exc:
            logger.debug("Hindsight bank list fetch failed: %s", exc)
        return self._bank_ids_cache or [self.bank_id]

    async def _select_bank_for_write(self, text: str, owner: Optional[str]) -> str:
        """Ask the utility model which bank best fits this memory."""
        banks = self._get_all_bank_ids()
        if len(banks) <= 1:
            return banks[0] if banks else self.bank_id
        try:
            from src.endpoint_resolver import resolve_endpoint
            from src.llm_core import llm_call_async
            url, model, headers = resolve_endpoint("utility", owner=owner)
            if not url:
                return self.bank_id
            bank_list = ", ".join(banks)
            chosen = await llm_call_async(
                url, model,
                [
                    {"role": "system", "content": _BANK_SELECT_WRITE_PROMPT},
                    {"role": "user", "content": f"Banks: {bank_list}\n\nMemory: {text}"},
                ],
                temperature=0.0,
                max_tokens=32,
                headers=headers,
                timeout=10,
            )
            chosen = chosen.strip().strip('"').strip("'").lower()
            if chosen in banks:
                logger.debug("Hindsight bank router → retain to '%s'", chosen)
                return chosen
        except Exception as exc:
            logger.debug("Hindsight bank selection for write failed: %s", exc)
        return self.bank_id

    def _select_banks_for_read(self, query: str, owner: Optional[str]) -> List[str]:
        """Ask the utility model which bank(s) to search for this query."""
        banks = self._get_all_bank_ids()
        if len(banks) <= 1:
            return banks
        try:
            from src.endpoint_resolver import resolve_endpoint
            from src.llm_core import llm_call
            url, model, headers = resolve_endpoint("utility", owner=owner)
            if not url:
                return banks
            bank_list = ", ".join(banks)
            resp = llm_call(
                url, model,
                [
                    {"role": "system", "content": _BANK_SELECT_READ_PROMPT},
                    {"role": "user", "content": f"Banks: {bank_list}\n\nQuery: {query}"},
                ],
                temperature=0.0,
                max_tokens=64,
                headers=headers,
                timeout=8,
            )
            chosen = [b.strip().strip('"').strip("'").lower() for b in resp.split(",")]
            valid = [b for b in chosen if b in banks]
            if valid:
                logger.debug("Hindsight bank router → recall from %s", valid)
                return valid
        except Exception as exc:
            logger.debug("Hindsight bank selection for read failed: %s", exc)
        return banks

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def check_health(self) -> bool:
        """Probe the server and update self.healthy. Called once on startup."""
        try:
            client = self._get_async_client()
            resp = await client.get(f"{self.base_url}/health")
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
            await client.put(
                f"{self.base_url}/v1/default/banks/{self.bank_id}",
                headers=self._headers,
                json={"name": self.bank_id},
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
        owner: Optional[str] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """Store a memory in the bank chosen by the utility model. Fire via asyncio.create_task()."""
        if not self.healthy:
            return False
        bank_id = await self._select_bank_for_write(text, owner)
        item: Dict[str, Any] = {"content": text}
        if context:
            item["context"] = context
        if metadata:
            item["metadata"] = metadata
        if timestamp:
            item["timestamp"] = timestamp
        try:
            client = self._get_async_client()
            resp = await client.post(
                f"{self.base_url}/v1/default/banks/{bank_id}/memories",
                headers=self._headers,
                json={"items": [item], "async": True},
            )
            return resp.is_success
        except Exception as exc:
            logger.debug("Hindsight retain failed: %s", exc)
            return False

    def recall(self, query: str, *, top_k: int = 5, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search the bank(s) chosen by the utility model and merge results."""
        if not self.healthy:
            return []
        bank_ids = self._select_banks_for_read(query, owner)
        seen: set = set()
        merged: List[Dict[str, Any]] = []
        for bid in bank_ids:
            try:
                resp = self._sync_client.post(
                    f"{self.base_url}/v1/default/banks/{bid}/memories/recall",
                    headers=self._headers,
                    json={"query": query, "budget": "low"},
                )
                if not resp.is_success:
                    continue
                for hit in _parse_recall(resp.json(), top_k):
                    key = hit["text"].strip().lower()
                    if key not in seen:
                        seen.add(key)
                        merged.append(hit)
            except Exception as exc:
                logger.debug("Hindsight recall failed for bank %s: %s", bid, exc)
        merged.sort(key=lambda h: h.get("score") or 0, reverse=True)
        return merged[:top_k]

    async def reflect(self, query: str, *, owner: Optional[str] = None) -> Optional[str]:
        """Synthesize memories via Hindsight reflect. Uses the utility-model-selected bank."""
        if not self.healthy:
            return None
        bank_id = await self._select_bank_for_write(query, owner)
        try:
            client = self._get_async_client()
            resp = await client.post(
                f"{self.base_url}/v1/default/banks/{bank_id}/reflect",
                headers=self._headers,
                json={"query": query},
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
