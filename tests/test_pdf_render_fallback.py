"""_process_pdf scanned-page fallback: low-text pages are rendered whole via
PyMuPDF and sent to the vision model (robust to embedded image encodings pypdf
can't decode), capped per PDF; the pypdf embedded-image path only runs when
PyMuPDF is absent, and VL error banners are never injected as PDF content."""
import io

import pypdf

import src.document_processor as dp


class _FakePage:
    def __init__(self, text="", images=None):
        self._text = text
        self.images = images or []

    def extract_text(self):
        return self._text


class _FakeReader:
    def __init__(self, pages):
        self.pages = pages


def _setup(monkeypatch, pages, fitz_available=True, vision_ready=True, vl_text="a scanned invoice"):
    calls = {"render": 0, "vl": 0}
    monkeypatch.setattr(pypdf, "PdfReader", lambda path: _FakeReader(pages))
    monkeypatch.setattr(dp, "_vision_ready", lambda owner=None: vision_ready)
    monkeypatch.setattr(dp, "_get_fitz", lambda: object() if fitz_available else None)

    def fake_render(path, page_index, dpi=150):
        calls["render"] += 1
        return f"/nonexistent/page-{page_index}.png"  # caller's unlink tolerates OSError

    def fake_vl(image_path, owner=None):
        calls["vl"] += 1
        return vl_text

    monkeypatch.setattr(dp, "_render_pdf_page_png", fake_render)
    monkeypatch.setattr(dp, "analyze_image_with_vl", fake_vl)
    return calls


def test_low_text_page_uses_full_page_render(monkeypatch):
    calls = _setup(monkeypatch, [_FakePage(text="")])
    result = dp._process_pdf("/fake.pdf")
    assert calls["render"] == 1
    assert "[Page 1 (rendered) text]: a scanned invoice" in result


def test_texty_page_skips_render(monkeypatch):
    calls = _setup(monkeypatch, [_FakePage(text="x" * 100)])
    result = dp._process_pdf("/fake.pdf")
    assert calls["render"] == 0
    assert "[Page 1 text]:" in result


def test_render_page_cap(monkeypatch):
    calls = _setup(monkeypatch, [_FakePage(text="") for _ in range(8)])
    dp._process_pdf("/fake.pdf")
    assert calls["render"] == dp._MAX_VL_RENDERED_PAGES


def test_vl_error_banner_not_injected(monkeypatch):
    _setup(monkeypatch, [_FakePage(text="")],
           vl_text="[No vision model configured — set one in Settings → Vision]")
    result = dp._process_pdf("/fake.pdf")
    assert "No vision model configured" not in result
    assert "no readable content" in result


def test_vision_not_ready_skips_all_vl_work(monkeypatch):
    calls = _setup(monkeypatch, [_FakePage(text="")], vision_ready=False)
    result = dp._process_pdf("/fake.pdf")
    assert calls["render"] == 0
    assert calls["vl"] == 0
    assert "no readable content" in result


def test_fitz_missing_falls_back_to_pypdf_images(monkeypatch):
    class _FakeEmbeddedImage:
        class image:
            @staticmethod
            def save(path, fmt):
                with open(path, "wb") as f:
                    f.write(b"png")

    calls = _setup(monkeypatch, [_FakePage(text="", images=[_FakeEmbeddedImage()])],
                   fitz_available=False)
    result = dp._process_pdf("/fake.pdf")
    assert calls["render"] == 0
    assert calls["vl"] == 1
    assert "[Page 1 image 1 text]: a scanned invoice" in result
