# src/pdf_capabilities.py
"""Which models accept native PDF (file) input.

The only authoritative, machine-readable source is OpenRouter's public model
catalog: models whose ``architecture.input_modalities`` contains ``"file"``
accept a PDF directly (Claude, Gemini, GPT-4o/5, Grok, some Mistral, …). The
catalog is fetched without auth, in a background thread so the model picker is
never blocked, and cached in-process. Matching is fuzzy enough to cover the
same model reached through a direct provider endpoint (``claude-sonnet-4-5``
vs OpenRouter's ``anthropic/claude-sonnet-4.5``)."""

import logging
import re
import threading
import time

import httpx

logger = logging.getLogger(__name__)

_CATALOG_URL = "https://openrouter.ai/api/v1/models"
_SUCCESS_TTL = 6 * 3600   # refresh the catalog a few times a day
_FAILURE_TTL = 3600       # offline installs: don't retry every picker open

_lock = threading.Lock()
_cache = {"ids": set(), "time": 0.0, "ttl": 0.0}
_fetch_inflight = False


def _id_variants(model_id: str) -> set:
    """Normalized spellings of a model id for cross-provider matching.

    Covers vendor prefixes (``anthropic/claude-x`` → ``claude-x``), OpenRouter
    ``:free``/``:beta`` suffixes, dots-vs-dashes version styles, and trailing
    Anthropic-style date stamps (``claude-sonnet-4-5-20250929``).
    """
    mid = (model_id or "").strip().lower()
    if not mid:
        return set()
    out = {mid, mid.rsplit("/", 1)[-1]}
    for v in list(out):
        out.add(v.split(":", 1)[0])
    for v in list(out):
        out.add(v.replace(".", "-"))
    for v in list(out):
        out.add(re.sub(r"-\d{8}$", "", v))
    return out


def _fetch_catalog_ids() -> None:
    global _fetch_inflight
    ids: set = set()
    ttl = _FAILURE_TTL
    try:
        r = httpx.get(_CATALOG_URL, timeout=10.0)
        if r.is_success:
            for m in (r.json().get("data") or []):
                modalities = ((m.get("architecture") or {}).get("input_modalities")) or []
                if "file" in modalities:
                    ids |= _id_variants(m.get("id"))
            ttl = _SUCCESS_TTL
            logger.info("PDF capability catalog loaded: %d id variants", len(ids))
        else:
            logger.warning("PDF capability catalog fetch failed: HTTP %s", r.status_code)
    except Exception as e:
        logger.warning("PDF capability catalog fetch failed: %s", e)
    with _lock:
        # Keep the last good set when a refresh fails.
        if ids or not _cache["ids"]:
            _cache["ids"] = ids
        _cache["time"] = time.time()
        _cache["ttl"] = ttl
        _fetch_inflight = False


def get_native_pdf_ids() -> set:
    """Current set of native-PDF model id variants; may be empty while the
    first background fetch is still running (badges appear on the next
    picker refresh)."""
    global _fetch_inflight
    with _lock:
        fresh = (time.time() - _cache["time"]) < _cache["ttl"]
        ids = _cache["ids"]
        if fresh or _fetch_inflight:
            return ids
        _fetch_inflight = True
    threading.Thread(target=_fetch_catalog_ids, daemon=True).start()
    return ids


def supports_native_pdf(model_id: str) -> bool:
    """True when the model accepts a PDF as native file input."""
    if not model_id:
        return False
    return not _id_variants(model_id).isdisjoint(get_native_pdf_ids())
