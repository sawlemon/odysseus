"""Native-PDF capability lookup: OpenRouter catalog ids must match the same
model reached through direct provider endpoints (vendor prefixes, dot/dash
version styles, date-stamped Anthropic ids, :free variants)."""
import time

import src.pdf_capabilities as pc


def _inject_catalog(monkeypatch, ids):
    variants = set()
    for i in ids:
        variants |= pc._id_variants(i)
    monkeypatch.setattr(pc, "_cache", {"ids": variants, "time": time.time(), "ttl": 9999})


def test_id_variants_cover_provider_spellings():
    v = pc._id_variants("anthropic/claude-sonnet-4.5")
    assert "claude-sonnet-4.5" in v
    assert "claude-sonnet-4-5" in v


def test_supports_native_pdf_matches(monkeypatch):
    _inject_catalog(monkeypatch, [
        "anthropic/claude-sonnet-4.5",
        "google/gemini-2.5-flash",
        "openai/gpt-4o",
    ])
    # Exact OpenRouter id
    assert pc.supports_native_pdf("anthropic/claude-sonnet-4.5")
    # Direct provider endpoint spellings
    assert pc.supports_native_pdf("claude-sonnet-4-5")
    assert pc.supports_native_pdf("claude-sonnet-4-5-20250929")  # date-stamped
    assert pc.supports_native_pdf("gemini-2.5-flash")
    assert pc.supports_native_pdf("gpt-4o")
    # OpenRouter :free variant of a listed model
    assert pc.supports_native_pdf("google/gemini-2.5-flash:free")
    # Non-file models
    assert not pc.supports_native_pdf("llama-3.3-70b-versatile")
    assert not pc.supports_native_pdf("qwen2-vl")
    assert not pc.supports_native_pdf("")


def test_empty_catalog_means_no_badges(monkeypatch):
    _inject_catalog(monkeypatch, [])
    assert not pc.supports_native_pdf("claude-sonnet-4-5")
