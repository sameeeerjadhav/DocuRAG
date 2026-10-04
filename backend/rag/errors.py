"""Domain errors for ingestion and question answering.

Routes translate these into HTTP responses. Keeping them out of FastAPI
lets the RAG modules be explained (and tested) without the web framework.
"""


class RagError(Exception):
    """Base class for expected failures the user can act on."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class UnsupportedFileTypeError(RagError):
    def __init__(self, filename: str):
        super().__init__(
            f"'{filename}' is not supported. Upload PDF or TXT files only.",
            400,
        )


class EmptyDocumentError(RagError):
    def __init__(self, filename: str):
        super().__init__(
            f"'{filename}' has no extractable text. Empty files and scanned "
            "image-only PDFs cannot be indexed (they need OCR first).",
            400,
        )


class FileTooLargeError(RagError):
    def __init__(self, filename: str, limit_mb: int):
        super().__init__(
            f"'{filename}' is larger than {limit_mb} MB. Split it and try again.",
            400,
        )


class NoDocumentsError(RagError):
    def __init__(self):
        super().__init__(
            "No documents are indexed for this session yet. Upload a PDF or TXT file first.",
            400,
        )


class SessionNotFoundError(RagError):
    def __init__(self, session_id: str):
        super().__init__(
            f"No indexed documents found for session {session_id}.",
            404,
        )


class ConfigurationError(RagError):
    def __init__(self, message: str):
        super().__init__(message, 500)


class ModelCallError(RagError):
    """Gemini (or the vector write that calls Gemini) failed."""

    def __init__(self, reason: str):
        short = " ".join(str(reason).split())[:300]
        super().__init__(
            "Gemini API request failed. Check GEMINI_API_KEY, the model names, "
            f"and your quota. Details: {short}",
            502,
        )
