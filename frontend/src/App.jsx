import { useEffect, useMemo, useState } from "react";
import { askQuestion, clearSession, listDocuments, uploadDocuments } from "./api.js";
import ChatPanel from "./components/ChatPanel.jsx";
import UploadPanel from "./components/UploadPanel.jsx";

const STORAGE_KEY = "docurag.session.v1";

function loadStoredSession() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { sessionId: null, messages: [] };
    const parsed = JSON.parse(raw);
    return {
      sessionId: typeof parsed.sessionId === "string" ? parsed.sessionId : null,
      messages: Array.isArray(parsed.messages) ? parsed.messages : [],
    };
  } catch {
    return { sessionId: null, messages: [] };
  }
}

function newId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  return `${Date.now()}-${Math.random()}`;
}

export default function App() {
  const stored = useMemo(() => loadStoredSession(), []);
  const [sessionId, setSessionId] = useState(stored.sessionId);
  const [messages, setMessages] = useState(stored.messages);
  const [documents, setDocuments] = useState(() => (stored.sessionId ? null : []));
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [chatError, setChatError] = useState("");

  useEffect(() => {
    if (!sessionId) {
      localStorage.removeItem(STORAGE_KEY);
      return;
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ sessionId, messages }));
  }, [sessionId, messages]);

  // Restore the document list for a session id saved in this browser.
  // Chat text is local; the chunks themselves live in Chroma on the server.
  useEffect(() => {
    if (!stored.sessionId) return undefined;
    let ignore = false;
    listDocuments(stored.sessionId)
      .then((data) => {
        if (ignore) return;
        setDocuments(data.documents || []);
      })
      .catch((error) => {
        if (ignore) return;
        if (error.status === 404) {
          setSessionId(null);
          setMessages([]);
          setDocuments([]);
          setUploadError("The saved session is no longer on the server. Upload a document to start a new one.");
          return;
        }
        setDocuments([]);
        setUploadError(error.message);
      });
    return () => {
      ignore = true;
    };
  }, [stored.sessionId]);

  const totalChunks = (documents || []).reduce((sum, doc) => sum + (doc.chunk_count || 0), 0);
  const canChat = Array.isArray(documents) && documents.length > 0 && totalChunks > 0 && !uploading;

  async function handleUpload(files) {
    const allowed = files.filter((file) => /\.(pdf|txt)$/i.test(file.name));
    if (!allowed.length) {
      setUploadError("Upload PDF or TXT files only.");
      return;
    }
    const skipped = allowed.length !== files.length;
    setUploadError("");
    setUploading(true);
    try {
      const data = await uploadDocuments(allowed, sessionId);
      setSessionId(data.session_id);
      setDocuments(data.documents || []);
      setUploadError(
        skipped ? "Indexed the PDF and TXT files. Other file types were skipped." : ""
      );
    } catch (error) {
      setUploadError(error.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleClear() {
    if (!sessionId) return;
    const confirmed = window.confirm("Clear this session and delete its indexed chunks?");
    if (!confirmed) return;
    setUploadError("");
    try {
      await clearSession(sessionId);
      setSessionId(null);
      setMessages([]);
      setDocuments([]);
      setChatError("");
    } catch (error) {
      setUploadError(error.message);
    }
  }

  async function handleSend(question) {
    if (!sessionId) {
      setChatError("Upload a document before asking a question.");
      return;
    }
    setChatError("");
    setMessages((current) => [...current, { id: newId(), role: "user", content: question }]);
    setSending(true);
    try {
      const data = await askQuestion(sessionId, question);
      setMessages((current) => [
        ...current,
        {
          id: newId(),
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
        },
      ]);
    } catch (error) {
      setChatError(error.message);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="min-h-full bg-[radial-gradient(ellipse_at_top,_#fbf8f3,_#e7e0d4)] text-stone-900">
      <header className="mx-auto flex max-w-6xl items-center justify-between px-4 py-5 lg:px-6">
        <div>
          <p className="font-serif text-3xl tracking-tight text-teal-950">DocuRAG</p>
          <p className="text-sm text-stone-500">Answers grounded in your documents</p>
        </div>
        <p className="hidden rounded-full bg-white/80 px-3 py-1 text-xs font-medium text-teal-900 shadow-sm sm:block">
          Retrieve, then generate
        </p>
      </header>
      <main className="mx-auto grid w-full max-w-6xl items-stretch gap-4 px-4 pb-6 lg:h-[calc(100vh-6.5rem)] lg:grid-cols-[22rem_minmax(0,1fr)] lg:px-6">
        <UploadPanel
          documents={documents || []}
          loadingDocs={documents === null}
          uploading={uploading}
          error={uploadError}
          sessionId={sessionId}
          totalChunks={totalChunks}
          onUpload={handleUpload}
          onClear={handleClear}
        />
        <ChatPanel
          messages={messages}
          sending={sending}
          error={chatError}
          canChat={canChat}
          onSend={handleSend}
        />
      </main>
    </div>
  );
}
