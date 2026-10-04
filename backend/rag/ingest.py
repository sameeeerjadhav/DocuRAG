"""Turn uploaded files into searchable chunks.

Pipeline (this is the "index" half of RAG):

1. Extract plain text from each PDF or TXT file.
2. Split that text into overlapping chunks.
3. Hand the chunks to Chroma, which calls Gemini to embed each one and
   stores the vector plus the original text and metadata.

The chat path never re-reads the file. It only searches this index.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag.errors import (
    EmptyDocumentError,
    FileTooLargeError,
    ModelCallError,
    RagError,
    UnsupportedFileTypeError,
)
from rag.vectorstore import (
    chroma_lock,
    collection_exists,
    delete_session,
    get_vectorstore,
    list_documents,
    new_session_id,
    parse_session_id,
)

logger = logging.getLogger(__name__)

# Measured in characters, not tokens. LangChain's default length function is
# len(). 1000 characters is roughly 200–250 tokens of English — about a
# paragraph or two. That is small enough to point at a specific fact and
# large enough to keep the surrounding sentence intact.
CHUNK_SIZE = 1000

# Overlap repeats the tail of one chunk at the head of the next. Without it,
# a sentence that straddles the cut would be split in half and might not
# match the question on either side. 150 characters is enough to cover a
# typical sentence without doubling the index.
CHUNK_OVERLAP = 150

MAX_FILES = 10
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_FILE_MB = MAX_FILE_BYTES // (1024 * 1024)
# A cap so one huge paste cannot fire thousands of embedding calls.
MAX_CHUNKS_PER_UPLOAD = 400

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    # Overlap is not "copy the last 150 characters" inside LangChain. The
    # splitter cuts the text into pieces, packs those pieces up to
    # chunk_size, then keeps pieces from the tail while they still fit in
    # chunk_overlap. A whole paragraph is longer than 150 characters, so if
    # we split on blank lines the overlap budget cannot hold one and the
    # next chunk starts with no shared text.
    #
    # Splitting on spaces makes each piece a word (shorter than 150). The
    # packer then repeats about 150 characters of words at the start of the
    # next chunk, and it will not cut a word in half. Newlines are not
    # separators here, so paragraph breaks stay in the chunk the model sees.
    separators=[" ", ""],
)


def ingest_files(session_id: str | None, files: list[tuple[str, bytes]]) -> dict:
    """Validate and chunk every file, then embed and store them.

    ``files`` is a list of ``(filename, raw_bytes)``. Passing an existing
    session id appends to that collection. Passing none starts a new session.

    Chunking happens before the collection is created, so a bad file (wrong
    type, empty, too large) fails the whole request and does not leave a
    partial index.
    """

    if session_id and str(session_id).strip():
        session_id = parse_session_id(session_id)
    else:
        session_id = new_session_id()

    if not files:
        raise RagError("Upload at least one PDF or TXT file.")
    if len(files) > MAX_FILES:
        raise RagError(f"Upload at most {MAX_FILES} files at a time.")

    documents: list[Document] = []
    for filename, data in files:
        documents.extend(chunk_file(filename, data))

    if len(documents) > MAX_CHUNKS_PER_UPLOAD:
        raise RagError(
            f"This upload would create {len(documents)} chunks, above the limit "
            f"of {MAX_CHUNKS_PER_UPLOAD}. Upload a shorter document."
        )

    with chroma_lock():
        return _store_chunks(session_id, documents)


def chunk_file(filename: str, data: bytes) -> list[Document]:
    """Extract text and split one file into Documents ready for embedding."""

    safe_name = _clean_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise UnsupportedFileTypeError(safe_name)
    if not data:
        raise EmptyDocumentError(safe_name)
    if len(data) > MAX_FILE_BYTES:
        raise FileTooLargeError(safe_name, MAX_FILE_MB)

    if suffix == ".pdf":
        pages = _extract_pdf(safe_name, data)
        file_type = "pdf"
    else:
        pages = [_extract_txt(safe_name, data)]
        file_type = "txt"

    # One LangChain Document per page (or the whole text file). The splitter
    # copies metadata onto every child chunk, so a chunk cut from page 4
    # still says page 4. Splitting per page — rather than gluing the PDF
    # into one string first — is what keeps page citations honest.
    parents: list[Document] = []
    for page in pages:
        # Uploads from Windows are CRLF. The splitter looks for "\n\n" and
        # "\n", which do not match if a "\r" is sitting between the newlines,
        # so paragraphs would be cut in the wrong places and overlap would
        # shrink to a single heading.
        text = page["text"].replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            continue
        metadata: dict = {"source": safe_name, "file_type": file_type}
        if page["page"] is not None:
            metadata["page"] = page["page"]
        parents.append(Document(page_content=text, metadata=metadata))

    if not parents:
        raise EmptyDocumentError(safe_name)

    splits = _SPLITTER.split_documents(parents)
    chunks: list[Document] = []
    for doc in splits:
        text = doc.page_content.strip()
        if not text:
            continue
        # chunk_index is 0-based within this file, in reading order.
        # It is assigned after splitting so the number matches what we store.
        doc.page_content = text
        doc.metadata = {**doc.metadata, "chunk_index": len(chunks)}
        chunks.append(doc)

    if not chunks:
        raise EmptyDocumentError(safe_name)
    return chunks


def _clean_filename(filename: str) -> str:
    # Browsers sometimes send a path. Keep only the last component so a
    # crafted name cannot look like a directory.
    name = Path(filename or "").name.replace("\x00", "").strip()
    if not name or name in {".", ".."}:
        raise RagError("One of the uploaded files is missing a name.")
    return name


def _extract_pdf(filename: str, data: bytes) -> list[dict]:
    from io import BytesIO

    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            # Some PDFs report themselves as encrypted but have an empty
            # user password. If a real password is required, extraction fails
            # below and we surface a readable error.
            reader.decrypt("")
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            pages.append({"page": number, "text": page.extract_text() or ""})
        return pages
    except RagError:
        raise
    except Exception as exc:
        raise RagError(f"Could not read '{filename}': {exc}") from exc


def _extract_txt(filename: str, data: bytes) -> dict:
    if b"\x00" in data:
        raise RagError(f"'{filename}' does not look like a text file.")
    text = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise RagError(f"Could not decode '{filename}' as text.")
    # TXT has no pages. Leaving page unset (instead of storing null) matters
    # because Chroma metadata values cannot be None.
    return {"page": None, "text": text}


def _store_chunks(session_id: str, documents: list[Document]) -> dict:
    """Embed ``documents`` and upsert them into this session's collection.

    ``add_documents`` is the step that calls the embedding model. We do not
    embed manually: the vector store embeds on write and again on query, so
    both sides are guaranteed to use the function we passed in.
    """

    existed = collection_exists(session_id)
    store = None
    ids: list[str] = []
    try:
        # Creating the store also constructs the embedding client, so a missing
        # API key fails here — after the collection may already exist. Roll
        # that empty collection back with the failed batch.
        store = get_vectorstore(session_id, create=True)
        ids = [str(uuid.uuid4()) for _ in documents]
        store.add_documents(documents, ids=ids)
    except RagError:
        _rollback(store, ids, session_id, existed)
        raise
    except Exception as exc:
        logger.exception("Embedding or Chroma write failed")
        _rollback(store, ids, session_id, existed)
        raise ModelCallError(exc) from exc

    logger.info("Indexed %s chunks for session %s", len(documents), session_id)
    listed = list_documents(session_id)
    return {
        "session_id": session_id,
        "chunks_indexed": len(documents),
        "total_chunks": listed["total_chunks"],
        "documents": listed["documents"],
    }


def _rollback(store, ids: list[str], session_id: str, existed: bool) -> None:
    """Drop a failed batch so a Gemini error does not leave a half index."""

    if store is not None and ids:
        try:
            store.delete(ids=ids)
        except Exception:
            logger.warning("Could not roll back chunk ids after a failed insert", exc_info=True)
    if not existed:
        try:
            delete_session(session_id)
        except Exception:
            logger.warning("Could not drop the new session after a failed insert", exc_info=True)
