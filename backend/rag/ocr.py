"""OCR for scanned PDFs.

pypdf only reads a text layer. A phone scan is a picture of a page, so that
layer is empty. This module renders those pages with PyMuPDF and asks Gemini
to transcribe the image. Nothing here needs a local Tesseract or Poppler install.
"""

from __future__ import annotations

import base64

from rag.errors import ModelCallError, RagError

# A long scan would otherwise send dozens of images. The first pages are
# enough for a resume or a form; the upload progress says when the rest was skipped.
MAX_OCR_PAGES = 15

_TRANSCRIBE = (
    "Transcribe every visible word on this scanned document page in reading order. "
    "Output only the transcription. If there is no text, output nothing."
)


def render_pages(data: bytes, filename: str, limit: int = MAX_OCR_PAGES) -> list[dict]:
    """Rasterize up to ``limit`` pages. Each item has page, png, and total."""

    try:
        import pymupdf
    except ImportError as exc:
        raise RagError(
            "This PDF has no text layer, and scanned-page reading needs the pymupdf package."
        ) from exc

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise RagError(f"Could not open '{filename}' for OCR: {exc}") from exc

    pages: list[dict] = []
    try:
        total = int(document.page_count)
        for index in range(min(total, limit)):
            pixmap = document[index].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
            pages.append({"page": index + 1, "png": pixmap.tobytes("png"), "total": total})
    finally:
        document.close()
    return pages


def transcribe_pdf(filename: str, data: bytes, progress=None) -> list[dict]:
    """Return ``{page, text}`` rows by sending each rendered page to Gemini."""

    rendered = render_pages(data, filename)
    if not rendered:
        return []

    total = int(rendered[0]["total"])
    if total > len(rendered) and progress:
        progress(f"Transcribing the first {len(rendered)} pages of {filename}")

    pages: list[dict] = []
    shown = min(total, MAX_OCR_PAGES)
    for item in rendered:
        if progress:
            progress(f"Transcribing page {item['page']} of {shown} in {filename}")
        pages.append({"page": item["page"], "text": _transcribe_png(item["png"])})
    return pages


def _transcribe_png(png: bytes) -> str:
    from langchain_core.messages import HumanMessage

    from rag.llm import get_chat_model
    from rag.retrieve import message_text

    encoded = base64.b64encode(png).decode("ascii")
    message = HumanMessage(
        content=[
            {"type": "text", "text": _TRANSCRIBE},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ]
    )
    try:
        reply = get_chat_model().invoke([message])
    except RagError:
        raise
    except Exception as exc:
        raise ModelCallError(exc) from exc
    return message_text(reply)
