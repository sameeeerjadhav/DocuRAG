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

export function uploadDocuments(files, sessionId) {
  const body = new FormData();
  for (const file of files) body.append("files", file);
  if (sessionId) body.append("session_id", sessionId);
  return request("/documents/upload", { method: "POST", body });
}

export function listDocuments(sessionId) {
  return request(`/documents/${sessionId}`);
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

export function askQuestion(sessionId, question) {
  return request(`/chat/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}
