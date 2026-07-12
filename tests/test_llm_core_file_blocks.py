"""Provider adaptation for PDF pass-through ``file`` content blocks:
OpenRouter gets the block plus the file-parser plugin, Anthropic gets a
converted ``document`` block, and everyone else gets the block stripped
(extracted text already lives in the text block)."""
import base64

import src.llm_core as llm_core


_B64_PDF = base64.b64encode(b"%PDF-1.4 fake").decode()
_FILE_BLOCK = {
    "type": "file",
    "file": {"filename": "scan.pdf", "file_data": f"data:application/pdf;base64,{_B64_PDF}"},
}


def _messages():
    return [
        {"role": "user", "content": [
            {"type": "text", "text": "read this\n\n[PDF content]: hello"},
            dict(_FILE_BLOCK),
        ]},
    ]


# ── content-block helpers ──

def test_convert_file_block_to_anthropic_document():
    converted = llm_core._convert_openai_content_to_anthropic(_messages()[0]["content"])
    doc_blocks = [b for b in converted if b.get("type") == "document"]
    assert len(doc_blocks) == 1
    assert doc_blocks[0]["source"] == {
        "type": "base64",
        "media_type": "application/pdf",
        "data": _B64_PDF,
    }
    # No raw "file" block may survive — Anthropic would 400 on it.
    assert not any(b.get("type") == "file" for b in converted)


def test_convert_unparsable_file_block_dropped():
    converted = llm_core._convert_openai_content_to_anthropic(
        [{"type": "file", "file": {"filename": "x.pdf", "file_data": "https://example.com/x.pdf"}}]
    )
    assert converted == []


def test_messages_have_file_blocks():
    assert llm_core._messages_have_file_blocks(_messages()) is True
    assert llm_core._messages_have_file_blocks([{"role": "user", "content": "hi"}]) is False
    assert llm_core._messages_have_file_blocks([]) is False


def test_strip_file_blocks_keeps_other_blocks():
    stripped = llm_core._strip_file_blocks(_messages())
    content = stripped[0]["content"]
    assert [b["type"] for b in content] == ["text"]
    # Original messages untouched (shallow copy of affected messages only).
    assert any(b.get("type") == "file" for b in _messages()[0]["content"])


def test_openrouter_pdf_plugins_engine_validation(monkeypatch):
    import src.settings as settings_mod
    monkeypatch.setattr(settings_mod, "get_setting", lambda key, default=None: "mistral-ocr")
    assert llm_core._openrouter_pdf_plugins() == [{"id": "file-parser", "pdf": {"engine": "mistral-ocr"}}]
    monkeypatch.setattr(settings_mod, "get_setting", lambda key, default=None: "bogus-engine")
    assert llm_core._openrouter_pdf_plugins() == [{"id": "file-parser", "pdf": {"engine": "native"}}]


# ── llm_call payload wiring ──

class _FakeResponse:
    is_success = True
    status_code = 200
    text = "ok"

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


def _call_and_capture(monkeypatch, url):
    captured = {}

    def fake_post(target_url, headers, json=None, timeout=None):
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr(llm_core, "httpx_post_kimi_aware", fake_post)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "_get_cached_response", lambda key: None)
    monkeypatch.setattr(llm_core, "_set_cached_response", lambda key, value: None)
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda target: False)
    llm_core.llm_call(url, "some-model", _messages())
    return captured["payload"]


def test_llm_call_openrouter_adds_plugins_and_keeps_file_block(monkeypatch):
    payload = _call_and_capture(monkeypatch, "https://openrouter.ai/api/v1/chat/completions")
    assert payload["plugins"] == [{"id": "file-parser", "pdf": {"engine": "native"}}]
    blocks = payload["messages"][-1]["content"]
    assert any(b.get("type") == "file" for b in blocks)


def test_llm_call_other_openai_provider_strips_file_block(monkeypatch):
    payload = _call_and_capture(monkeypatch, "https://api.groq.com/openai/v1/chat/completions")
    assert "plugins" not in payload
    blocks = payload["messages"][-1]["content"]
    assert not any(b.get("type") == "file" for b in blocks)
    assert any(b.get("type") == "text" for b in blocks)
