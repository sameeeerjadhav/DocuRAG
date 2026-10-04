"""Checks that do not need the running API, plus an optional retrieval eval.

From the backend directory:

    .\\.venv\\Scripts\\python.exe -m eval.run_eval

Overlap, duplicate-file skipping, rate limiting, and PDF rendering always run.
They use a temporary Chroma directory and a fake embedding function, so they
do not touch the app's index and do not call Gemini.

If GEMINI_API_KEY is set, the script also indexes samples/employee-handbook.txt
with the real embedding model and checks that the expected phrases come back.
That part is skipped, with a message, when the key is missing. It is not a CI gate.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_ROOT = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _safe(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = os.getenv(name) or ""
        if key:
            text = text.replace(key, "[key]")
    return text[:300]


def _shared_prefix(left: str, right: str) -> int:
    shared = 0
    for size in range(min(len(left), len(right)), 0, -1):
        if left.endswith(right[:size]):
            shared = size
            break
    return shared


def _use_temp_chroma() -> str:
    path = tempfile.mkdtemp(prefix="docurag-eval-")
    os.environ["CHROMA_PERSIST_DIR"] = path
    import rag.vectorstore as vectorstore

    vectorstore._client = None
    if "docurag-eval-" not in str(vectorstore.persist_dir()):
        raise RuntimeError("Refusing to run the eval against the app's Chroma directory.")
    return path


def _check_overlap() -> None:
    from rag.ingest import chunk_file

    data = (_ROOT / "samples" / "employee-handbook.txt").read_bytes()
    chunks = chunk_file("employee-handbook.txt", data)
    if len(chunks) < 2:
        raise AssertionError(f"expected overlapping chunks, got {len(chunks)}")
    shared = _shared_prefix(chunks[0].page_content, chunks[1].page_content)
    if shared < 100:
        raise AssertionError(f"chunk overlap was {shared} characters")
    print(f"overlap ok ({len(chunks)} chunks, {shared} shared characters)")


def _check_dedup_and_remove() -> None:
    from langchain_core.embeddings import Embeddings

    import rag.llm as llm
    from rag.ingest import ingest_files
    from rag.vectorstore import remove_document

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[0.2, 0.1, 0.4] for _ in texts]

        def embed_query(self, text):
            return [0.2, 0.1, 0.4]

    original = llm.get_embeddings
    llm.get_embeddings = lambda: FakeEmbeddings()
    try:
        data = (_ROOT / "samples" / "employee-handbook.txt").read_bytes()
        first = ingest_files(None, [("employee-handbook.txt", data)])
        if first["chunks_indexed"] < 2:
            raise AssertionError("first index did not store chunks")
        second = ingest_files(first["session_id"], [("employee-handbook.txt", data)])
        if second["chunks_indexed"] != 0:
            raise AssertionError("duplicate upload was indexed again")
        if not second["skipped"] or second["skipped"][0]["reason"] != "already indexed":
            raise AssertionError(f"duplicate was not skipped: {second['skipped']}")
        if second["total_chunks"] != first["total_chunks"]:
            raise AssertionError("duplicate upload changed the chunk count")

        updated = data + b"\nThe snack cupboard is restocked every Monday morning with fruit.\n"
        third = ingest_files(first["session_id"], [("employee-handbook.txt", updated)])
        if third["chunks_indexed"] < 1:
            raise AssertionError("a changed file was not reindexed")
        if third["total_chunks"] != third["chunks_indexed"]:
            raise AssertionError("replacing a file left the old chunks in place")

        removed = remove_document(first["session_id"], "employee-handbook.txt")
        if removed["total_chunks"] != 0:
            raise AssertionError("remove file left chunks behind")
        print("dedup and remove ok")
    finally:
        llm.get_embeddings = original


def _check_rate_limit() -> None:
    from rag.errors import RateLimitError
    from rag.ratelimit import check_rate

    check_rate("eval:rate", 2)
    check_rate("eval:rate", 2)
    try:
        check_rate("eval:rate", 2)
    except RateLimitError:
        print("rate limit ok")
        return
    raise AssertionError("the third request was not limited")


def _check_page_render() -> None:
    import pymupdf

    from rag.ocr import render_pages

    document = pymupdf.open()
    document.new_page()
    data = document.tobytes()
    document.close()
    pages = render_pages(data, "blank.pdf")
    if not pages or not pages[0]["png"].startswith(b"\x89PNG"):
        raise AssertionError("scanned-page render did not produce a PNG")
    print("pdf render ok")


def _check_retrieval() -> int:
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        print("GEMINI_API_KEY is not set. Skipping the handbook retrieval check.")
        return 0

    from rag.ingest import ingest_files
    from rag.retrieve import retrieve_documents

    data = (_ROOT / "samples" / "employee-handbook.txt").read_bytes()
    indexed = ingest_files(None, [("employee-handbook.txt", data)])
    questions = json.loads((_BACKEND / "eval" / "questions.json").read_text(encoding="utf-8"))
    for item in questions:
        docs = retrieve_documents(indexed["session_id"], item["question"])
        blob = "\n".join(doc.page_content for doc in docs)
        missing = [phrase for phrase in item["expect"] if phrase not in blob]
        if missing:
            raise AssertionError(
                f"{item['question']} did not retrieve {missing}. Got {len(docs)} chunks."
            )
        print(f"retrieved ok: {item['question']}")

    followup = retrieve_documents(
        indexed["session_id"],
        "How often is that paid?",
        [
            {"role": "user", "content": "What is the laptop stipend?"},
            {"role": "assistant", "content": "The laptop stipend is $1,200 every three years."},
        ],
    )
    blob = "\n".join(doc.page_content for doc in followup)
    if "$1,200" not in blob and "three years" not in blob:
        raise AssertionError("follow-up query did not retrieve the stipend passage")
    print("follow-up retrieval ok")
    return 0


def main() -> int:
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    path = _use_temp_chroma()
    try:
        _check_overlap()
        _check_dedup_and_remove()
        _check_rate_limit()
        _check_page_render()
        _use_temp_chroma()
        return _check_retrieval()
    except Exception as exc:
        print(f"eval failed: {_safe(exc)}")
        return 1
    finally:
        shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(os.environ.get("CHROMA_PERSIST_DIR", ""), ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
