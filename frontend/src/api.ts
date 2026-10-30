// Types mirror the FastAPI schemas in app/schemas. Every call goes through the dev
// server's /api proxy, so the browser never needs CORS.

export type DocumentStatus = "uploaded" | "processing" | "indexed" | "failed";

export interface DocumentVersion {
  version: number;
  content_hash: string;
  status: DocumentStatus;
  error: string | null;
  mime_type: string | null;
  file_size: number | null;
  page_count: number | null;
  chunk_count: number | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentOut {
  id: number;
  filename: string;
  source_type: "file" | "url";
  source_url: string | null;
  status: DocumentStatus;
  active_version: number | null;
  created_at: string;
  updated_at: string;
  versions: DocumentVersion[];
}

export interface UploadResponse {
  document: DocumentOut;
  duplicate: boolean;
}

export interface DeleteResponse {
  id: number;
  deleted: boolean;
  chunks_removed: number;
}

export interface ConversationSummary {
  id: number;
  title: string | null;
  created_at: string;
  message_count: number;
  last_activity: string;
}

export interface Citation {
  document_id: number;
  filename: string;
  page: number | null;
  chunk_id: string;
  source_url: string | null;
  /** Missing on messages stored before snippets were added. */
  text?: string | null;
}

export interface TraceItem {
  chunk_id: string;
  document_id: number;
  version: number;
  filename: string;
  page: number | null;
  chunk_index: number;
  score: number;
  text: string;
  in_prompt: boolean;
  used_in_answer: boolean;
}

export interface Timings {
  rewrite_ms: number;
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
}

export interface ChatResponse {
  conversation_id: number;
  answer: string;
  citations: Citation[];
  insufficient_evidence: boolean;
  search_query: string;
  /** Chunks retrieval asked for: the question's top_k, else the TOP_K setting. */
  top_k: number;
  history_used: number;
  retrieval_trace: TraceItem[];
  timings: Timings | null;
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  created_at: string;
}

export interface Health {
  status: "ok" | "degraded";
  qdrant: string;
  mysql: string;
}

export interface Config {
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  min_score: number;
  history_limit: number;
  max_upload_mb: number;
  embedding_model: string;
  llm_model: string;
  rewrite_model: string;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    // Some error responses (a failed upload, 422) still carry a useful body.
    public body: unknown = null,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, init);
  } catch {
    throw new ApiError(0, "Cannot reach the API. Is the backend running?");
  }
  const text = await response.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!response.ok) {
    throw new ApiError(response.status, errorMessage(response.status, body), body);
  }
  // Every successful response is wrapped as {"data": ...}.
  return (body as { data: T }).data;
}

function errorMessage(status: number, body: unknown): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    // FastAPI validation errors: a list of {msg, loc}.
    if (Array.isArray(detail)) return detail.map((d) => d?.msg ?? String(d)).join("; ");
  }
  return `Request failed (${status})`;
}

function json(method: string, data?: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: data === undefined ? undefined : JSON.stringify(data),
  };
}

function fileForm(file: File): RequestInit {
  const form = new FormData();
  form.append("file", file);
  return { body: form };
}

/** Upload and update answer 422 when indexing fails; the body still has the document. */
async function uploadRequest(path: string, method: string, file: File): Promise<UploadResponse> {
  try {
    return await request<UploadResponse>(path, { method, ...fileForm(file) });
  } catch (e) {
    const failed = e instanceof ApiError && e.status === 422 ? dataOf(e.body) : null;
    if (isUploadResponse(failed)) {
      return failed;
    }
    throw e;
  }
}

function dataOf(body: unknown): unknown {
  return body && typeof body === "object" && "data" in body ? (body as { data: unknown }).data : null;
}

function isUploadResponse(body: unknown): body is UploadResponse {
  return !!body && typeof body === "object" && "document" in body;
}

export const api = {
  health: () => request<Health>("/health"),
  config: () => request<Config>("/config"),
  documents: () => request<DocumentOut[]>("/documents"),
  upload: (file: File) => uploadRequest("/documents", "POST", file),
  replace: (id: number, file: File) => uploadRequest(`/documents/${id}`, "PUT", file),
  remove: (id: number) => request<DeleteResponse>(`/documents/${id}`, { method: "DELETE" }),
  conversations: () => request<ConversationSummary[]>("/conversations"),
  createConversation: () => request<{ id: number }>("/conversations", json("POST")),
  history: (id: number) =>
    request<{ conversation_id: number; messages: Message[] }>(`/chat/${id}/history`),
  ask: (id: number, question: string, topK?: number) =>
    request<ChatResponse>(`/chat/${id}`, json("POST", { question, top_k: topK })),
};
