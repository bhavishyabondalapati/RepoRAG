// Typed client for the RepoRAG FastAPI backend.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Mode = "hybrid_rerank" | "hybrid" | "dense" | "bm25";

export type RepoStatus = {
  repo_id: string;
  repo: string;
  status: "queued" | "cloning" | "chunking" | "embedding" | "ready" | "failed";
  progress: number;
  message: string;
  error: string | null;
  commit: string | null;
  files_indexed: number;
  chunks: number;
};

export type Citation = {
  ref: number;
  path: string;
  start_line: number;
  end_line: number;
  language: string;
  symbol: string | null;
  snippet: string;
  score: number;
  url: string;
};

export type AskResponse = {
  answer: string;
  insufficient: boolean;
  citations: Citation[];
  mode: Mode;
  latency_ms: number;
};

export type Health = { status: string; llm_configured: boolean };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      // non-JSON error body; keep the status text
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  index: (url: string, force = false) =>
    request<RepoStatus>("/api/repos", { method: "POST", body: JSON.stringify({ url, force }) }),
  status: (repoId: string) => request<RepoStatus>(`/api/repos/${encodeURIComponent(repoId)}`),
  ask: (repoId: string, question: string, mode: Mode) =>
    request<AskResponse>(`/api/repos/${encodeURIComponent(repoId)}/ask`, {
      method: "POST",
      body: JSON.stringify({ question, mode }),
    }),
};
