import SourceList from "./SourceList.jsx";

const INSUFFICIENT = "I don't have enough information";

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const declined = !isUser && message.content.trim() === INSUFFICIENT;

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-teal-900 px-4 py-2.5 text-sm leading-relaxed text-white shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[92%] rounded-2xl rounded-bl-md border border-stone-200 bg-white px-4 py-3 text-sm leading-relaxed text-stone-800 shadow-sm">
        <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-teal-800">
          DocuRAG
        </p>
        <p className="whitespace-pre-wrap">
          {message.content}
          {message.pending ? (
            <span className="ml-0.5 inline-block h-3.5 w-0.5 translate-y-0.5 animate-pulse bg-teal-800" />
          ) : null}
        </p>
        {declined ? (
          <p className="mt-2 text-xs text-stone-500">
            Nothing in the indexed documents was relevant, so DocuRAG did not guess.
          </p>
        ) : null}
        <SourceList sources={message.sources} />
      </div>
    </div>
  );
}
