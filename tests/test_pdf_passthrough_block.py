"""PDF pass-through: build_user_content emits an OpenRouter-shaped ``file``
content block for attached PDFs so file-capable providers (OpenRouter,
Anthropic) can read scanned/image pages directly. The block is gated by the
``pdf_passthrough_enabled`` setting and the ``pdf_passthrough_max_mb`` size
cap, and extracted text always stays inline as the safety net."""
import base64

import src.document_processor as dp


class _FakeUploadHandler:
    def is_image_file(self, name, mime):
        return False

    def is_audio_file(self, name, mime):
        return False

    def is_document_file(self, name, mime):
        return True

    def _inside_upload_dir(self, path):
        return True


def _build(tmp_path, monkeypatch, settings, pdf_bytes=b"%PDF-1.4 fake", size=None):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(pdf_bytes)
    monkeypatch.setattr(dp, "_load_vl_settings", lambda: settings)
    monkeypatch.setattr(dp, "_process_pdf", lambda path, owner=None: "\n\n[PDF content]:\n\n[Page 1 text]:\nhello")
    resolved = {"fid1": {"path": str(pdf_path), "mime": "application/pdf", "name": "scan.pdf"}}
    if size is not None:
        resolved["fid1"]["size"] = size
    return dp.build_user_content(
        text="what does this say",
        attachment_ids=["fid1"],
        upload_dir=str(tmp_path),
        upload_handler=_FakeUploadHandler(),
        session_id=None,
        resolved_uploads=resolved,
    )


def _file_blocks(content):
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "file"]


def test_pdf_emits_file_block_when_enabled(tmp_path, monkeypatch):
    content = _build(tmp_path, monkeypatch, {"pdf_passthrough_enabled": True, "pdf_passthrough_max_mb": 10})
    blocks = _file_blocks(content)
    assert len(blocks) == 1
    file_data = blocks[0]["file"]["file_data"]
    assert file_data.startswith("data:application/pdf;base64,")
    assert base64.b64decode(file_data.split(",", 1)[1]) == b"%PDF-1.4 fake"
    assert blocks[0]["file"]["filename"] == "scan.pdf"
    # Extracted text stays inline as the fallback for non-file providers.
    assert "[Page 1 text]:" in content[0]["text"]


def test_pdf_only_message_keeps_list_shape(tmp_path, monkeypatch):
    # A file block alone must count as media — otherwise build_user_content
    # collapses the list to a plain string and drops the block.
    content = _build(tmp_path, monkeypatch, {"pdf_passthrough_enabled": True})
    assert isinstance(content, list)


def test_no_file_block_when_disabled(tmp_path, monkeypatch):
    content = _build(tmp_path, monkeypatch, {"pdf_passthrough_enabled": False})
    assert _file_blocks(content) == []
    # Without media the content collapses to plain text with the extraction.
    assert isinstance(content, str)
    assert "[Page 1 text]:" in content


def test_no_file_block_when_oversize(tmp_path, monkeypatch):
    content = _build(
        tmp_path,
        monkeypatch,
        {"pdf_passthrough_enabled": True, "pdf_passthrough_max_mb": 1},
        size=2 * 1024 * 1024,
    )
    assert _file_blocks(content) == []


def test_bad_max_mb_setting_falls_back_to_default(tmp_path, monkeypatch):
    content = _build(
        tmp_path,
        monkeypatch,
        {"pdf_passthrough_enabled": True, "pdf_passthrough_max_mb": "not-a-number"},
    )
    assert len(_file_blocks(content)) == 1
