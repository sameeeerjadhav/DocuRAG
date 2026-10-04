const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readError(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail
        .map((item) => item.msg || JSON.stringify(item))
        .join(" ");
    }
  } catch {
    // The body was not JSON. Fall through to the status code.
  }
  return `Request failed (${response.status}).`;
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, options);
  } catch {
    throw new ApiError(
      `Cannot reach the DocuRAG API at ${API_URL}. Start the backend and try again.`,
      0
    );
  }
  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  if (response.status === 204) return null;
  return response.json();
}

async function openStream(path, options) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, options);
  } catch {
    throw new ApiError(
      `Cannot reach the DocuRAG API at ${API_URL}. Start the backend and try again.`,
      0
    );
  }
  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  return response;
}

async function readSse(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      if (!frame.trim()) continue;
      let event = "message";
      const dataLines = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      const data = dataLines.length ? JSON.parse(dataLines.join("\n")) : {};
      onEvent(event, data);
    }
  }
}

export async function uploadDocuments(files, sessionId, onStatus) {
  const body = new FormData();
  for (const file of files) body.append("files", file);
  if (sessionId) body.append("session_id", sessionId);
  const response = await openStream("/documents/upload/stream", { method: "POST", body });
  let result = null;
  let streamError = null;
  await readSse(response, (event, data) => {
    if (event === "status" && onStatus) onStatus(data.message || "");
    if (event === "result") result = data;
    if (event === "error") {
      streamError = new ApiError(data.detail || "Upload failed.", data.status || 400);
    }
  });
  if (streamError) throw streamError;
  if (!result) throw new ApiError("Upload ended without a result.", 500);
  return result;
}

export function listDocuments(sessionId) {
  return request(`/documents/${sessionId}`);
}

export function removeDocument(sessionId, filename) {
  const params = new URLSearchParams({ filename });
  return request(`/documents/${sessionId}/file?${params}`, { method: "DELETE" });
}

export async function clearSession(sessionId) {
  let response;
  try {
    response = await fetch(`${API_URL}/documents/${sessionId}`, { method: "DELETE" });
  } catch {
    throw new ApiError(
      `Cannot reach the DocuRAG API at ${API_URL}. Start the backend and try again.`,
      0
    );
  }
  // A missing session is already clear. Treat that as success.
  if (response.status === 404) return null;
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  return response.json();
}

export function askQuestion(sessionId, question, history = []) {
  return request(`/chat/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });
}

export async function streamQuestion(sessionId, question, history, onEvent) {
  const response = await openStream(`/chat/${sessionId}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });
  let streamError = null;
  await readSse(response, (event, data) => {
    if (event === "error") {
      streamError = new ApiError(data.detail || "The question failed.", data.status || 502);
      return;
    }
    onEvent(event, data);
  });
  if (streamError) throw streamError;
}
