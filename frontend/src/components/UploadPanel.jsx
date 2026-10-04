import { useRef, useState } from "react";

function extensionOf(filename) {
  const parts = filename.split(".");
  return parts.length > 1 ? parts.pop().toUpperCase() : "FILE";
}

export default function UploadPanel({
  documents,
  loadingDocs,
  uploading,
  uploadStatus,
  removing,
  error,
  note,
  sessionId,
  totalChunks,
  onUpload,
  onRemove,
  onClear,
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function takeFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length || uploading) return;
    onUpload(files);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <aside className="flex min-h-[28rem] flex-col overflow-hidden rounded-3xl border border-white/80 bg-white/90 p-4 shadow-card lg:h-full lg:min-h-0">
      <div className="px-1">
        <h2 className="text-sm font-semibold text-stone-900">Documents</h2>
        <p className="mt-1 text-sm leading-relaxed text-stone-500">
          PDF or TXT. Files are split into chunks, embedded, and stored for this session only.
        </p>
      </div>

      <button
        type="button"
        disabled={uploading}
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          if (!uploading) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          takeFiles(event.dataTransfer.files);
        }}
        className={`mt-4 rounded-2xl border border-dashed px-4 py-8 text-center transition ${
          dragging
            ? "border-teal-700 bg-teal-50"
            : "border-stone-300 bg-stone-50 hover:border-teal-700 hover:bg-white"
        } disabled:cursor-wait disabled:opacity-70`}
      >
        <span className="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-teal-900 text-white">
          <svg viewBox="0 0 20 20" className="h-5 w-5" aria-hidden="true">
            <path
              d="M10 3.5v9M6.5 7 10 3.5 13.5 7M4.5 13.5v1.75A1.25 1.25 0 0 0 5.75 16.5h8.5a1.25 1.25 0 0 0 1.25-1.25V13.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
        <p className="mt-3 text-sm font-medium text-stone-800">
          {uploading ? uploadStatus || "Indexing documents…" : "Drop files here or browse"}
        </p>
        <p className="mt-1 text-xs text-stone-500">
          {uploading ? "This can take a minute for a scan or a large file." : "Up to 10 files, 20 MB each"}
        </p>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt,application/pdf,text/plain"
        multiple
        className="hidden"
        onChange={(event) => takeFiles(event.target.files)}
      />

      {error ? (
        <p role="alert" className="mt-3 rounded-xl bg-red-50 px-3 py-2 text-sm text-red-800">
          {error}
        </p>
      ) : null}
      {note ? (
        <p className="mt-3 rounded-xl bg-amber-50 px-3 py-2 text-sm text-amber-950">{note}</p>
      ) : null}

      <div className="mt-4 flex items-baseline justify-between px-1">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-stone-400">Indexed</h3>
        <p className="text-xs text-stone-400">
          {loadingDocs ? "Loading…" : `${documents.length} file${documents.length === 1 ? "" : "s"} · ${totalChunks} chunks`}
        </p>
      </div>

      <ul className="mt-2 flex-1 space-y-2 overflow-y-auto pr-1">
        {loadingDocs ? (
          <li className="rounded-2xl bg-stone-50 px-3 py-3 text-sm text-stone-500">Restoring session…</li>
        ) : null}
        {!loadingDocs && documents.length === 0 ? (
          <li className="rounded-2xl border border-stone-100 px-3 py-4 text-sm text-stone-500">
            Nothing indexed yet. Try <span className="font-medium text-stone-700">samples/employee-handbook.txt</span>.
          </li>
        ) : null}
        {documents.map((doc) => (
          <li
            key={doc.filename}
            className="flex items-center justify-between gap-3 rounded-2xl border border-stone-100 bg-stone-50 px-3 py-2.5"
          >
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-stone-800">{doc.filename}</p>
              <p className="text-xs text-stone-500">
                {doc.chunk_count} chunk{doc.chunk_count === 1 ? "" : "s"}
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <span className="rounded-md bg-white px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-stone-500">
                {extensionOf(doc.filename)}
              </span>
              <button
                type="button"
                onClick={() => onRemove(doc.filename)}
                disabled={uploading || removing}
                className="text-xs font-medium text-stone-500 hover:text-red-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Remove
              </button>
            </div>
          </li>
        ))}
      </ul>

      {sessionId ? (
        <p className="mt-3 truncate px-1 font-mono text-[11px] text-stone-400" title={sessionId}>
          Session {sessionId}
        </p>
      ) : null}

      <button
        type="button"
        onClick={onClear}
        disabled={!sessionId || uploading}
        className="mt-3 w-full rounded-xl border border-stone-200 px-3 py-2 text-sm font-medium text-stone-600 transition hover:border-red-200 hover:bg-red-50 hover:text-red-700 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-stone-200 disabled:hover:bg-transparent disabled:hover:text-stone-600"
      >
        Clear session
      </button>
    </aside>
  );
}
