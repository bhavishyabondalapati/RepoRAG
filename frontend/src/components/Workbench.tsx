"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api, type AskResponse, type Citation, type Health, type Mode, type RepoStatus } from "@/lib/api";
import { AnswerBody } from "./AnswerBody";
import { IndexProgress } from "./IndexProgress";
import { SourcePanel } from "./SourcePanel";
import styles from "./Workbench.module.css";

type Turn = {
  id: number;
  question: string;
  mode: Mode;
  state: "loading" | "done" | "error";
  response?: AskResponse;
  error?: string;
};

const EXAMPLES = [
  "https://github.com/pallets/itsdangerous",
  "https://github.com/sindresorhus/ky",
];

const SUGGESTIONS: Record<string, string[]> = {
  "pallets/itsdangerous": [
    "How does TimestampSigner decide that a signature has expired?",
    "How does key rotation work when verifying signatures?",
    "Does itsdangerous encrypt the payload?",
  ],
  "sindresorhus/ky": [
    "Which status codes are retried by default?",
    "How is the request timeout implemented?",
    "How does ky.extend merge options?",
  ],
};
const GENERIC_SUGGESTIONS = [
  "What is the main entry point of this project?",
  "How is configuration loaded?",
  "How are errors handled?",
];

const MODES: { value: Mode; label: string }[] = [
  { value: "hybrid_rerank", label: "hybrid + rerank" },
  { value: "hybrid", label: "hybrid" },
  { value: "dense", label: "dense" },
  { value: "bm25", label: "bm25" },
];

export function Workbench() {
  const [health, setHealth] = useState<Health | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const [url, setUrl] = useState("");
  const [job, setJob] = useState<RepoStatus | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("hybrid_rerank");
  const [active, setActive] = useState<Citation | null>(null);
  const nextId = useRef(1);
  const transcriptEnd = useRef<HTMLDivElement>(null);
  const sourceRef = useRef<HTMLElement>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setBackendDown(true));
  }, []);

  // Poll indexing progress until the job settles.
  useEffect(() => {
    if (!job || job.status === "ready" || job.status === "failed") return;
    const timer = setTimeout(() => {
      api.status(job.repo_id).then(setJob).catch((e: Error) => setIndexError(e.message));
    }, 700);
    return () => clearTimeout(timer);
  }, [job]);

  useEffect(() => {
    transcriptEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const startIndex = useCallback(async (target: string, force = false) => {
    setIndexError(null);
    setTurns([]);
    setActive(null);
    try {
      setJob(await api.index(target.trim(), force));
    } catch (e) {
      setJob(null);
      setIndexError((e as Error).message);
    }
  }, []);

  const ask = useCallback(
    async (q: string) => {
      if (!job || job.status !== "ready" || !q.trim()) return;
      const id = nextId.current++;
      setTurns((t) => [...t, { id, question: q.trim(), mode, state: "loading" }]);
      setQuestion("");
      try {
        const response = await api.ask(job.repo_id, q.trim(), mode);
        setTurns((t) => t.map((x) => (x.id === id ? { ...x, state: "done", response } : x)));
        if (response.citations.length) setActive(response.citations[0]);
      } catch (e) {
        setTurns((t) => t.map((x) => (x.id === id ? { ...x, state: "error", error: (e as Error).message } : x)));
      }
    },
    [job, mode],
  );

  const showCitation = (c: Citation) => {
    setActive(c);
    if (window.matchMedia("(max-width: 959px)").matches) {
      sourceRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const ready = job?.status === "ready";
  const suggestions = job ? (SUGGESTIONS[job.repo] ?? GENERIC_SUGGESTIONS) : [];

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.brand}>
          <span className={`${styles.wordmark} mono`}>reporag</span>
          <span className={styles.tagline}>Answers about a codebase, with the lines to prove it.</span>
        </div>
        <StatusPill health={health} backendDown={backendDown} />
      </header>

      <form
        className={styles.repoBar}
        onSubmit={(e) => {
          e.preventDefault();
          if (url.trim()) startIndex(url);
        }}
      >
        <label htmlFor="repo-url" className={`${styles.prompt} mono`}>
          repo
        </label>
        <input
          id="repo-url"
          className={`${styles.repoInput} mono`}
          placeholder="https://github.com/owner/repo"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          autoComplete="off"
          spellCheck={false}
        />
        <button className={styles.primary} type="submit" disabled={!url.trim() || (!!job && !ready && job.status !== "failed")}>
          Index
        </button>
      </form>

      {!job && !indexError && (
        <p className={styles.examples}>
          Try{" "}
          {EXAMPLES.map((ex, i) => (
            <span key={ex}>
              <button
                type="button"
                className={`${styles.linkButton} mono`}
                onClick={() => {
                  setUrl(ex);
                  startIndex(ex);
                }}
              >
                {ex.replace("https://github.com/", "")}
              </button>
              {i < EXAMPLES.length - 1 ? " or " : ""}
            </span>
          ))}
          . Public repos up to 50 MB.
        </p>
      )}

      {indexError && <p className={styles.error}>{indexError}</p>}
      {job && <IndexProgress job={job} onRetry={() => startIndex(`https://github.com/${job.repo}`, true)} />}

      <main className={styles.workspace}>
        <section className={styles.chat} aria-label="Conversation">
          <div className={styles.transcript}>
            {turns.length === 0 && (
              <div className={styles.empty}>
                {ready ? (
                  <>
                    <p className={styles.emptyLead}>Ask anything about {job.repo}.</p>
                    <ul className={styles.suggestions}>
                      {suggestions.map((s) => (
                        <li key={s}>
                          <button type="button" className={styles.suggestion} onClick={() => ask(s)}>
                            {s}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </>
                ) : (
                  <p className={styles.emptyLead}>
                    Index a repository to start. Every answer cites file paths and line ranges, and RepoRAG
                    says so when the repo doesn&apos;t contain the answer.
                  </p>
                )}
              </div>
            )}

            {turns.map((t) => (
              <article key={t.id} className={styles.turn}>
                <p className={styles.question}>
                  <span className={`${styles.caret} mono`} aria-hidden>
                    ›
                  </span>
                  {t.question}
                </p>
                {t.state === "loading" && <p className={`${styles.thinking} mono`}>retrieving · reranking · answering</p>}
                {t.state === "error" && <p className={styles.error}>{t.error}</p>}
                {t.state === "done" && t.response && (
                  <AnswerBody response={t.response} active={active} onCite={showCitation} />
                )}
              </article>
            ))}
            <div ref={transcriptEnd} />
          </div>

          <form
            className={styles.composer}
            onSubmit={(e) => {
              e.preventDefault();
              ask(question);
            }}
          >
            <textarea
              className={styles.textarea}
              placeholder={ready ? "Ask about the code…  (Enter to send, Shift+Enter for a new line)" : "Waiting for an indexed repo"}
              value={question}
              disabled={!ready}
              rows={2}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(question);
                }
              }}
            />
            <div className={styles.composerFoot}>
              <div className={styles.modes} role="radiogroup" aria-label="Retrieval mode">
                {MODES.map((m) => (
                  <button
                    key={m.value}
                    type="button"
                    role="radio"
                    aria-checked={mode === m.value}
                    className={`${styles.mode} mono`}
                    onClick={() => setMode(m.value)}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
              <button className={styles.primary} type="submit" disabled={!ready || !question.trim()}>
                Ask
              </button>
            </div>
          </form>
        </section>

        <aside className={styles.source} ref={sourceRef} aria-label="Cited source">
          <SourcePanel citation={active} />
        </aside>
      </main>
    </div>
  );
}

function StatusPill({ health, backendDown }: { health: Health | null; backendDown: boolean }) {
  if (backendDown) return <span className={`${styles.pill} ${styles.pillBad} mono`}>backend offline</span>;
  if (!health) return <span className={`${styles.pill} mono`}>connecting…</span>;
  return health.llm_configured ? (
    <span className={`${styles.pill} ${styles.pillOk} mono`}>backend ready</span>
  ) : (
    <span className={`${styles.pill} ${styles.pillBad} mono`} title="Set GEMINI_API_KEY in .env">
      llm key missing
    </span>
  );
}
