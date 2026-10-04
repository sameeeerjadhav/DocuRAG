import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { isInsufficient, plainAnswer } from "../answerText.js";
import SourceList from "./SourceList.jsx";

const markdownComponents = {
  h1: ({ node, ...props }) => (
    <h2 className="mb-2 mt-4 font-serif text-2xl leading-tight text-stone-950 first:mt-0" {...props} />
  ),
  h2: ({ node, ...props }) => (
    <h3 className="mb-2 mt-4 text-lg font-semibold leading-snug text-stone-950 first:mt-0" {...props} />
  ),
  h3: ({ node, ...props }) => (
    <h4 className="mb-1.5 mt-3 text-base font-semibold text-stone-950 first:mt-0" {...props} />
  ),
  p: ({ node, ...props }) => <p className="mb-3 last:mb-0" {...props} />,
  ul: ({ node, ...props }) => <ul className="mb-3 list-disc space-y-1 pl-5 last:mb-0" {...props} />,
  ol: ({ node, ...props }) => <ol className="mb-3 list-decimal space-y-1 pl-5 last:mb-0" {...props} />,
  li: ({ node, ...props }) => <li className="pl-1" {...props} />,
  strong: ({ node, ...props }) => <strong className="font-semibold text-stone-950" {...props} />,
  em: ({ node, ...props }) => <em {...props} />,
  a: ({ node, ...props }) => (
    <a className="font-medium text-teal-800 underline decoration-teal-800/40 underline-offset-2" target="_blank" rel="noreferrer" {...props} />
  ),
  blockquote: ({ node, ...props }) => (
    <blockquote className="mb-3 border-l-2 border-teal-800 pl-3 text-stone-600 last:mb-0" {...props} />
  ),
  hr: () => <hr className="my-4 border-stone-200" />,
  pre: ({ node, ...props }) => (
    <pre className="mb-3 overflow-x-auto rounded-xl bg-stone-900 p-3 text-[13px] leading-6 text-stone-100 last:mb-0" {...props} />
  ),
  code: ({ node, inline, className, children, ...props }) =>
    inline ? (
      <code className="rounded bg-stone-100 px-1 py-0.5 font-mono text-[0.85em] text-stone-900" {...props}>
        {children}
      </code>
    ) : (
      <code className={`font-mono text-[0.85em] ${className || ""}`} {...props}>
        {children}
      </code>
    ),
  table: ({ node, ...props }) => (
    <div className="mb-3 overflow-x-auto last:mb-0">
      <table className="w-full border-collapse text-left text-sm" {...props} />
    </div>
  ),
  th: ({ node, ...props }) => <th className="border-b border-stone-200 px-2 py-1.5 font-semibold text-stone-950" {...props} />,
  td: ({ node, ...props }) => <td className="border-b border-stone-100 px-2 py-1.5 align-top" {...props} />,
};

function answerKind(content) {
  const text = plainAnswer(content);
  if (!text) return null;
  if (isInsufficient(content)) return "missing";
  if (text.startsWith("This is not stated in the document")) return "inferred";
  return "stated";
}

function displayMarkdown(content, pending) {
  if (!pending) return content;
  let text = content;
  if (((text.match(/\*\*/g) || []).length) % 2) text += "**";
  if (((text.match(/`/g) || []).length) % 2) text += "`";
  return text;
}

const KIND_LABEL = {
  stated: "From the file",
  inferred: "Inferred",
  missing: "Not in the file",
};

const KIND_CLASS = {
  stated: "bg-teal-50 text-teal-900",
  inferred: "bg-amber-50 text-amber-950",
  missing: "bg-stone-100 text-stone-500",
};

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const declined = !isUser && isInsufficient(message.content);
  const kind = isUser || message.pending ? null : answerKind(message.content);
  const [copied, setCopied] = useState(false);

  async function copyAnswer() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

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
      <div className="w-full max-w-[42rem] rounded-2xl rounded-bl-md border border-stone-200 bg-white px-5 py-4 text-stone-800 shadow-sm">
        <div className="mb-1 flex items-center justify-between gap-3">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-teal-800">DocuRAG</p>
          <div className="flex items-center gap-1.5">
            {kind ? (
              <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${KIND_CLASS[kind]}`}>
                {KIND_LABEL[kind]}
              </span>
            ) : null}
            {message.content && !message.pending ? (
              <button
                type="button"
                onClick={copyAnswer}
                className="rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-stone-400 hover:bg-stone-100 hover:text-stone-700"
              >
                {copied ? "Copied" : "Copy"}
              </button>
            ) : null}
          </div>
        </div>
        <div className="max-w-none text-[15px] leading-7 text-stone-800">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {displayMarkdown(message.content, message.pending)}
          </ReactMarkdown>
          {message.pending ? (
            <span className="ml-0.5 inline-block h-4 w-0.5 translate-y-0.5 animate-pulse bg-teal-800" />
          ) : null}
        </div>
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
