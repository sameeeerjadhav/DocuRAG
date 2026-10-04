"""Gemini chat and embedding clients.

Both objects come from ``langchain-google-genai``. The API key is read from
``GEMINI_API_KEY`` (what this project documents) and also copied to
``GOOGLE_API_KEY``, which older LangChain releases look up first.

Why two models?
- Embeddings turn text into a vector so we can search by meaning.
- The chat model writes the answer. It never sees the raw file, only the
  few chunks retrieval picked out.
"""

from __future__ import annotations

import os
from pathlib import Path

from rag.errors import ConfigurationError

# docurag/.env is gitignored. Load it so a local uvicorn picks up GEMINI_API_KEY
# without the key having to sit on the command line.
try:
    from dotenv import load_dotenv

    _ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(_ROOT / ".env")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

DEFAULT_CHAT_MODEL = "gemini-3.8-flash"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"


def require_api_key() -> str:
    """Return the Gemini key or raise a clear configuration error.

    The key stays on the server. The React app never receives it.
    """

    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise ConfigurationError(
            "GEMINI_API_KEY is not set. Add it to the backend environment "
            "before uploading or chatting."
        )
    # Recent langchain-google-genai checks GEMINI_API_KEY, then GOOGLE_API_KEY.
    os.environ["GEMINI_API_KEY"] = key
    os.environ.setdefault("GOOGLE_API_KEY", key)
    return key


def _model_fields(cls: type) -> set[str]:
    fields = getattr(cls, "model_fields", None) or getattr(cls, "__fields__", {})
    return set(fields)


def _with_api_key(cls: type, kwargs: dict) -> dict:
    """Pass the key under whichever constructor field this version exposes."""

    key = require_api_key()
    fields = _model_fields(cls)
    if "api_key" in fields:
        kwargs["api_key"] = key
    elif "google_api_key" in fields:
        kwargs["google_api_key"] = key
    return kwargs


def get_embeddings():
    """Embedding model used both when indexing and when searching.

    The same model must be used for both directions. Vectors from different
    models do not live in a comparable space, so nearest-neighbor search
    would return nonsense. ``gemini-embedding-001`` is the text embedding
    model that still supports retrieval task types: LangChain tags document
    chunks as ``RETRIEVAL_DOCUMENT`` and the question as ``RETRIEVAL_QUERY``.
    Override with ``GEMINI_EMBED_MODEL`` if your key needs a newer id
    (for example ``gemini-embedding-2``).
    """

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model_name = os.getenv("GEMINI_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip()
    kwargs = _with_api_key(GoogleGenerativeAIEmbeddings, {"model": model_name})
    return GoogleGenerativeAIEmbeddings(**kwargs)


def get_chat_model():
    """Flash chat model that writes the grounded answer.

    Temperature is 0 so the model sticks to the excerpts instead of
    embellishing them. ``gemini-3.8-flash`` is the default. New API keys
    cannot call ``gemini-2.5-flash``. Set ``GEMINI_CHAT_MODEL`` to override.
    """

    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = os.getenv("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL).strip()
    fields = _model_fields(ChatGoogleGenerativeAI)
    extra = {}
    # Gemini 2.5 Flash spends extra tokens "thinking" unless this is 0.
    # Grounded QA does not need that pass. Newer flash models use a different
    # thinking control, so only the 2.5 family gets this knob.
    if "thinking_budget" in fields and "2.5" in model_name:
        extra["thinking_budget"] = 0
    kwargs = _with_api_key(
        ChatGoogleGenerativeAI,
        {"model": model_name, "temperature": 0, **extra},
    )
    return ChatGoogleGenerativeAI(**kwargs)
