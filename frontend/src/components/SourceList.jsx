// Citations for one assistant turn. These are the chunks the retriever
// passed into the prompt, in rank order (best match first), not a separate
// citation the model invented.

export default function SourceList({ sources }) {
  if (!sources?.length) return null;

  return (
    <details className="mt-3 rounded-xl border border-stone-200 bg-stone-50 px-3 py-2">
      <summary className="flex cursor-pointer list-none items-center gap-2 text-xs font-semibold uppercase tracking-wide text-stone-500">
        <svg
          className="source-chevron h-3 w-3 transition-transform"
          viewBox="0 0 12 12"
          aria-hidden="true"
        >
          <path
            d="M4 2.5 8 6 4 9.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        Sources
        <span className="font-medium normal-case tracking-normal text-stone-400">
          {sources.length} retrieved excerpt{sources.length === 1 ? "" : "s"}
        </span>
      </summary>
      <ul className="mt-2 space-y-2">
        {sources.map((source, index) => (
          <li
            key={`${source.filename}-${source.chunk_index}-${index}`}
            className="rounded-lg bg-white px-3 py-2 text-sm text-stone-700 shadow-sm"
          >
            <p className="text-xs font-semibold text-teal-900">
              {source.filename}
              {source.page ? <span className="text-stone-400"> · page {source.page}</span> : null}
              <span className="text-stone-400"> · chunk index {source.chunk_index}</span>
            </p>
            <p className="mt-1 leading-relaxed text-stone-600">{source.snippet}</p>
          </li>
        ))}
      </ul>
    </details>
  );
}
