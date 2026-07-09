"""HindsightMemoryProvider — bridges HindsightClient into the MemoryProvider ABC.

list_memories() and delete() are intentional no-ops: the native memory.json
store owns listing, editing, deleting, pinning, and timeline. Hindsight is the
write + semantic-recall target for new memories; native is the UI index and
offline fallback.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.memory_provider import MemoryProvider, MemoryRecord, MemorySearchHit

if TYPE_CHECKING:
    from src.hindsight_client import HindsightClient

logger = logging.getLogger(__name__)


class HindsightMemoryProvider(MemoryProvider):
    """MemoryProvider backed by the local Hindsight server."""

    provider_id = "hindsight"
    display_name = "Hindsight"

    def __init__(self, client: "HindsightClient") -> None:
        self._client = client

    @property
    def enabled(self) -> bool:  # type: ignore[override]
        return self._client.healthy

    async def initialize(self) -> None:
        await self._client.check_health()
        if self._client.healthy:
            await self._client.ensure_bank()
            logger.info(
                "HindsightMemoryProvider ready (bank=%s, base_url=%s)",
                self._client.bank_id,
                self._client.base_url,
            )

    async def shutdown(self) -> None:
        self._client.close()

    async def remember(
        self,
        text: str,
        *,
        owner: Optional[str] = None,
        session_id: Optional[str] = None,
        category: str = "fact",
        source: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        meta: Dict[str, Any] = {"category": category, "source": source}
        if owner:
            meta["owner"] = owner
        if session_id:
            meta["session_id"] = session_id
        if metadata:
            meta.update(metadata)
        await self._client.retain(text, metadata=meta)
        return MemoryRecord(
            id="",
            text=text,
            timestamp=int(time.time()),
            category=category,
            source=source,
            owner=owner,
            session_id=session_id,
            metadata=meta,
        )

    async def recall(
        self,
        query: str,
        *,
        owner: Optional[str] = None,
        top_k: int = 5,
    ) -> List[MemorySearchHit]:
        raw = self._client.recall(query, top_k=top_k)
        return [
            MemorySearchHit(
                memory=MemoryRecord(id="", text=hit["text"]),
                provider_id=self.provider_id,
                score=hit.get("score"),
            )
            for hit in raw
        ]

    async def list_memories(
        self,
        *,
        owner: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemoryRecord]:
        # Native provider owns listing; return [] to signal no-op.
        return []

    async def delete(self, memory_id: str, *, owner: Optional[str] = None) -> bool:
        # Native provider owns deletion; return False.
        return False
