"""DocuRAG HTTP API.

Endpoints
---------
POST   /documents/upload                 Index one or more PDF/TXT files.
POST   /documents/upload/stream          Same upload, with progress events.
GET    /documents/{session_id}           List what that session has indexed.
DELETE /documents/{session_id}           Drop that session's Chroma collection.
DELETE /documents/{session_id}/file      Drop one filename from the session.
POST   /chat/{session_id}                Ask a question grounded in those chunks.
POST   /chat/{session_id}/stream         Same answer, token by token.

Run from the ``backend`` directory:

    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from rag.errors import RagError
from rag.ingest import ingest_files
from rag.ratelimit import check_rate
from rag.retrieve import answer_question, iter_answer
from rag.vectorstore import delete_session, list_documents, parse_session_id, persist_dir, remove_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("docurag")

# Vite (dev) and the nginx container (docker compose) both need to be listed.
# Override with CORS_ORIGINS if the UI is served from somewhere else.
_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

# Per client address, per minute. High enough for a person clicking around,
# low enough that a loop cannot hammer Gemini.
_UPLOADS_PER_MINUTE = 10
_CHATS_PER_MINUTE = 30


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("DocuRAG API ready. Chroma directory: %s", persist_dir())
    yield


app = FastAPI(
    title="DocuRAG",
    description="Upload documents, then ask questions answered only from those documents.",
    version="1.0.0",
    lifespan=lifespan,
)

_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", ",".join(_DEFAULT_ORIGINS)).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RagError)
async def handle_rag_error(_request, exc: RagError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int


class SkippedFile(BaseModel):
    filename: str
    reason: str


class UploadResponse(BaseModel):
    session_id: str
    chunks_indexed: int
    total_chunks: int
    documents: list[DocumentInfo]
    skipped: list[SkippedFile] = []


class SessionDocumentsResponse(BaseModel):
    session_id: str
    total_chunks: int
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    session_id: str
    deleted: bool


class ChatTurn(BaseModel):
    role: str
    content: str = Field(default="", max_length=2000)

    @field_validator("role")
    @classmethod
    def known_role(cls, value: str) -> str:
        if value not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant.")
        return value


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Question cannot be empty.")
        return text


class SourceCitation(BaseModel):
    filename: str
    chunk_index: int
    page: int | None = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


def _client_key(request: Request, action: str) -> str:
    host = request.client.host if request.client else "unknown"
    return f"{action}:{host}"


async def _read_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        payloads.append((upload.filename or "untitled", await upload.read()))
    return payloads


def _history(body: ChatRequest) -> list[dict]:
    return [{"role": turn.role, "content": turn.content} for turn in body.history]


async def _iter_sse(factory):
    """Run a sync event generator on a thread and yield server-sent events."""

    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for item in factory():
                asyncio.run_coroutine_threadsafe(events.put(item), loop).result()
        except RagError as exc:
            payload = ("error", {"detail": exc.message, "status": exc.status_code})
            asyncio.run_coroutine_threadsafe(events.put(payload), loop).result()
        except Exception:
            logger.exception("Stream failed")
            payload = ("error", {"detail": "The request failed.", "status": 500})
            asyncio.run_coroutine_threadsafe(events.put(payload), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(events.put(None), loop).result()

    threading.Thread(target=run, daemon=True).start()
    while True:
        item = await events.get()
        if item is None:
            break
        event, data = item
        yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse(factory):
    return StreamingResponse(
        _iter_sse(factory),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _upload_events(session_id: str | None, payloads: list[tuple[str, bytes]]):
    updates: queue.Queue = queue.Queue()

    def progress(message: str) -> None:
        updates.put(("status", {"message": message}))

    def work():
        try:
            result = ingest_files(session_id, payloads, progress=progress)
            updates.put(("result", result))
        except RagError as exc:
            updates.put(("error", {"detail": exc.message, "status": exc.status_code}))
        except Exception:
            logger.exception("Upload failed")
            updates.put(("error", {"detail": "Upload failed.", "status": 500}))
        finally:
            updates.put(None)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = updates.get()
        if item is None:
            break
        yield item


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"name": "DocuRAG", "health": "/health", "docs": "/docs"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(
    request: Request,
    files: list[UploadFile] = File(description="One or more PDF or TXT files."),
    session_id: str | None = Form(
        default=None,
        description="Existing session to append to. Omit to start a new session.",
    ),
):
    check_rate(_client_key(request, "upload"), _UPLOADS_PER_MINUTE)
    payloads = await _read_uploads(files)
    # Chroma and the Gemini client are synchronous. Run them off the event
    # loop so one long embedding job does not freeze every other request.
    return await asyncio.to_thread(ingest_files, session_id, payloads)


@app.post("/documents/upload/stream")
async def upload_documents_stream(
    request: Request,
    files: list[UploadFile] = File(description="One or more PDF or TXT files."),
    session_id: str | None = Form(default=None),
):
    check_rate(_client_key(request, "upload"), _UPLOADS_PER_MINUTE)
    payloads = await _read_uploads(files)
    return _sse(lambda: _upload_events(session_id, payloads))


@app.get("/documents/{session_id}", response_model=SessionDocumentsResponse)
async def get_documents(session_id: str):
    return await asyncio.to_thread(list_documents, session_id)


@app.delete("/documents/{session_id}/file", response_model=SessionDocumentsResponse)
async def remove_file(session_id: str, filename: str):
    return await asyncio.to_thread(remove_document, session_id, filename)


@app.delete("/documents/{session_id}", response_model=DeleteResponse)
async def clear_documents(session_id: str):
    session_id = parse_session_id(session_id)
    await asyncio.to_thread(delete_session, session_id)
    return {"session_id": session_id, "deleted": True}


@app.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(session_id: str, body: ChatRequest, request: Request):
    check_rate(_client_key(request, "chat"), _CHATS_PER_MINUTE)
    return await asyncio.to_thread(answer_question, session_id, body.question, _history(body))


@app.post("/chat/{session_id}/stream")
async def chat_stream(session_id: str, body: ChatRequest, request: Request):
    check_rate(_client_key(request, "chat"), _CHATS_PER_MINUTE)
    history = _history(body)
    return _sse(lambda: iter_answer(session_id, body.question, history))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
