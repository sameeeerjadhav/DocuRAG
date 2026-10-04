"""Persistent ChromaDB access, one collection per chat session.

A session is just a UUID. Its chunks live in a Chroma collection named
``sess_<uuid hex>`` under ``CHROMA_PERSIST_DIR`` (default ``backend/chroma_data``).

Why a collection per session, instead of one shared collection plus a
metadata filter?
- Similarity search is automatically limited to that user's files.
- "Clear session" is a single ``delete_collection`` call.
- A metadata filter on a shared collection is a valid alternative, but it is
  easier to get wrong (forget the filter once, and you leak another
  session's text into the prompt).

Chroma persists to disk, so restarting the API does not wipe the index.
The browser only remembers the session id; it does not store vectors.
"""

from __future__ import annotations

import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

# Chroma phones home unless this is set before the package is imported.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings

from rag.errors import RagError, SessionNotFoundError

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_client: chromadb.ClientAPI | None = None
# One process-wide client, and one lock. Chroma's local SQLite store
# misbehaves if two PersistentClients open the same path, and uvicorn is
# pinned to a single worker for the same reason (see the Dockerfile).
_lock = threading.RLock()


def persist_dir() -> Path:
    raw = os.getenv("CHROMA_PERSIST_DIR")
    if not raw:
        return _BACKEND_ROOT / "chroma_data"
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        path = persist_dir()
        path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


@contextmanager
def chroma_lock():
    with _lock:
        yield


def new_session_id() -> str:
    return str(uuid.uuid4())


def parse_session_id(session_id: str) -> str:
    try:
        return str(uuid.UUID(str(session_id).strip()))
    except ValueError as exc:
        raise RagError("session_id must be a valid UUID.", 400) from exc


def collection_name_for(session_id: str) -> str:
    """Chroma names must be 3–63 chars and alphanumeric plus ``_`` or ``-``."""

    return f"sess_{uuid.UUID(session_id).hex}"


def _missing_collection(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    return "notfound" in name or "does not exist" in text or "not found" in text


def open_collection(session_id: str, *, create: bool):
    """Return the raw Chroma collection, or None if it does not exist.

    New collections use cosine distance. Embeddings are compared by
    direction (the angle between vectors), which is what retrieval
    embeddings are trained for. Chroma's default is L2 (straight-line
    distance); cosine is a better match here. ``similarity_search`` still
    just asks for the nearest neighbors under whatever space the collection
    was created with.
    """

    client = get_client()
    name = collection_name_for(session_id)
    try:
        return client.get_collection(name)
    except Exception as exc:
        if not _missing_collection(exc):
            raise
        if not create:
            return None
        return client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )


def get_vectorstore(session_id: str, *, create: bool):
    """LangChain wrapper around this session's collection.

    We always pass our Gemini embedding function. If we forgot, Chroma would
    silently embed with its built-in local model and queries would not match
    the vectors we stored.
    """

    from langchain_chroma import Chroma

    from rag.llm import get_embeddings

    collection = open_collection(session_id, create=create)
    if collection is None:
        return None
    return Chroma(
        client=get_client(),
        collection_name=collection.name,
        embedding_function=get_embeddings(),
    )


def _list_documents(session_id: str) -> dict:
    collection = open_collection(session_id, create=False)
    if collection is None:
        raise SessionNotFoundError(session_id)

    # ids always come back; asking only for metadatas avoids pulling every
    # chunk body into memory just to draw the sidebar.
    data = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    order: list[str] = []
    for meta in data.get("metadatas") or []:
        meta = meta or {}
        name = str(meta.get("source") or "unknown")
        if name not in counts:
            order.append(name)
            counts[name] = 0
        counts[name] += 1

    documents = [
        {"filename": name, "chunk_count": counts[name]}
        for name in order
    ]
    return {
        "session_id": session_id,
        "total_chunks": sum(counts.values()),
        "documents": documents,
    }


def list_documents(session_id: str) -> dict:
    session_id = parse_session_id(session_id)
    with chroma_lock():
        return _list_documents(session_id)


def delete_session(session_id: str) -> None:
    session_id = parse_session_id(session_id)
    client = get_client()
    name = collection_name_for(session_id)
    with chroma_lock():
        try:
            client.delete_collection(name)
        except Exception as exc:
            if _missing_collection(exc):
                raise SessionNotFoundError(session_id) from exc
            raise


def collection_exists(session_id: str) -> bool:
    return open_collection(session_id, create=False) is not None


def chunk_count(session_id: str) -> int:
    """How many chunks are stored for this session. 0 if it does not exist."""

    collection = open_collection(session_id, create=False)
    if collection is None:
        return 0
    return int(collection.count())
