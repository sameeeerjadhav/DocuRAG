"""DocuRAG HTTP API.

Endpoints
---------
POST   /documents/upload          Index one or more PDF/TXT files.
GET    /documents/{session_id}    List what that session has indexed.
DELETE /documents/{session_id}    Drop that session's Chroma collection.
POST   /chat/{session_id}         Ask a question grounded in those chunks.

Run from the ``backend`` directory:

    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from rag.errors import RagError
from rag.ingest import ingest_files
from rag.retrieve import answer_question
from rag.vectorstore import delete_session, list_documents, parse_session_id, persist_dir

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


class UploadResponse(BaseModel):
    session_id: str
    chunks_indexed: int
    total_chunks: int
    documents: list[DocumentInfo]


class SessionDocumentsResponse(BaseModel):
    session_id: str
    total_chunks: int
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    session_id: str
    deleted: bool


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"name": "DocuRAG", "health": "/health", "docs": "/docs"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(description="One or more PDF or TXT files."),
    session_id: str | None = Form(
        default=None,
        description="Existing session to append to. Omit to start a new session.",
    ),
):
    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        payloads.append((upload.filename or "untitled", await upload.read()))
    # Chroma and the Gemini client are synchronous. Run them off the event
    # loop so one long embedding job does not freeze every other request.
    return await asyncio.to_thread(ingest_files, session_id, payloads)


@app.get("/documents/{session_id}", response_model=SessionDocumentsResponse)
async def get_documents(session_id: str):
    return await asyncio.to_thread(list_documents, session_id)


@app.delete("/documents/{session_id}", response_model=DeleteResponse)
async def clear_documents(session_id: str):
    session_id = parse_session_id(session_id)
    await asyncio.to_thread(delete_session, session_id)
    return {"session_id": session_id, "deleted": True}


@app.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(session_id: str, body: ChatRequest):
    return await asyncio.to_thread(answer_question, session_id, body.question)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
