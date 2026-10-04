import { useEffect, useRef, useState } from "react";
import MessageBubble from "./MessageBubble.jsx";

function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="rounded-2xl rounded-bl-md border border-stone-200 bg-white px-4 py-3 shadow-sm">
        <p className="text-[11px] font-semibold uppercase tracking-wide text-teal-800">DocuRAG</p>
        <p className="mt-1 flex items-center gap-1 text-sm text-stone-500">
          Searching your documents
          <span className="ml-1 inline-flex gap-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-teal-800 [animation-delay:-0.3s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-teal-800 [animation-delay:-0.15s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-teal-800" />
          </span>
        </p>
      </div>
    </div>
  );
}

export default function ChatPanel({ messages, sending, error, canChat, onSend }) {
  const [draft, setDraft] = useState("");
  const bottomRef = useRef(null);
  const areaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, sending]);

  function submit(event) {
    event?.preventDefault();
    const question = draft.trim();
    if (!question || sending || !canChat) return;
    setDraft("");
    if (areaRef.current) areaRef.current.style.height = "auto";
    onSend(question);
  }

  return (
    <section className="flex min-h-[32rem] flex-col overflow-hidden rounded-3xl border border-white/80 bg-white/90 shadow-card lg:h-full lg:min-h-0">
      <div className="border-b border-stone-100 px-5 py-4">
        <h2 className="text-sm font-semibold text-stone-900">Ask a question</h2>
        <p className="mt-1 text-sm text-stone-500">
          Answers use your documents as evidence. DocuRAG can summarize and reason from them, and it labels anything the file does not actually say.
        </p>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-5">
        {messages.length === 0 && !sending ? (
          <div className="mx-auto mt-10 max-w-sm text-center">
            <p className="font-serif text-2xl text-stone-800">What do your documents say?</p>
            <p className="mt-2 text-sm leading-relaxed text-stone-500">
              Upload a file on the left, then ask in everyday language. Each answer includes the filename and the matching snippet.
            </p>
          </div>
        ) : null}
        {messages.map((message) =>
          message.role === "assistant" && !message.content && !(message.sources || []).length ? null : (
            <MessageBubble key={message.id} message={message} />
          )
        )}
        {sending &&
        messages[messages.length - 1]?.role === "assistant" &&
        !messages[messages.length - 1]?.content &&
        !(messages[messages.length - 1]?.sources || []).length ? (
          <TypingIndicator />
        ) : null}
        <div ref={bottomRef} />
      </div>

      {error ? (
        <p role="alert" className="mx-4 mb-2 rounded-xl bg-red-50 px-3 py-2 text-sm text-red-800">
          {error}
        </p>
      ) : null}

      <form onSubmit={submit} className="border-t border-stone-100 p-3">
        <div className="flex items-end gap-2 rounded-2xl border border-stone-200 bg-stone-50 p-2 focus-within:border-teal-800 focus-within:bg-white">
          <label htmlFor="question" className="sr-only">
            Question
          </label>
          <textarea
            id="question"
            ref={areaRef}
            rows={1}
            value={draft}
            disabled={!canChat || sending}
            placeholder={canChat ? "Ask something that appears in your documents" : "Upload a document to ask a question"}
            onChange={(event) => {
              setDraft(event.target.value);
              event.target.style.height = "auto";
              event.target.style.height = `${Math.min(event.target.scrollHeight, 144)}px`;
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            className="max-h-36 min-h-[44px] flex-1 resize-none bg-transparent px-2 py-2.5 text-sm text-stone-800 outline-none placeholder:text-stone-400 disabled:cursor-not-allowed"
          />
          <button
            type="submit"
            disabled={!canChat || sending || !draft.trim()}
            className="rounded-xl bg-teal-900 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-teal-800 disabled:cursor-not-allowed disabled:bg-stone-300"
          >
            {sending ? "Sending…" : "Send"}
          </button>
        </div>
        <p className="mt-2 px-1 text-[11px] text-stone-400">Enter to send · Shift+Enter for a new line</p>
      </form>
    </section>
  );
}
