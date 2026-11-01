import ReactMarkdown from "react-markdown";

import type { AskResponse, Citation } from "@/lib/api";
import styles from "./Workbench.module.css";

// Turn "[1]" / "[1, 3]" markers into links the markdown renderer hands back to us.
const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

function linkCitations(text: string): string {
  return text.replace(CITATION, (_, group: string) =>
    group
      .split(",")
      .map((n) => `[${n.trim()}](#cite-${n.trim()})`)
      .join(""),
  );
}

function shortPath(path: string): string {
  const parts = path.split("/");
  return parts.length > 2 ? `…/${parts.slice(-2).join("/")}` : path;
}

type Props = {
  response: AskResponse;
  active: Citation | null;
  onCite: (c: Citation) => void;
};

export function AnswerBody({ response, active, onCite }: Props) {
  if (response.insufficient) {
    return (
      <div className={styles.insufficient}>
        <p>{response.answer}</p>
        <p className={styles.insufficientNote}>
          The retrieved code didn&apos;t support an answer, so RepoRAG declined rather than guess.
        </p>
      </div>
    );
  }

  const byRef = new Map(response.citations.map((c) => [c.ref, c]));
  const isActive = (c: Citation) => active?.url === c.url;

  return (
    <div className={styles.answer}>
      <div className={styles.prose}>
        <ReactMarkdown
          components={{
            a: ({ href, children }) => {
              const ref = href?.startsWith("#cite-") ? Number(href.slice(6)) : NaN;
              const citation = byRef.get(ref);
              if (!citation) return <span>{children}</span>;
              return (
                <button
                  type="button"
                  className={`${styles.cite} mono`}
                  data-active={isActive(citation)}
                  onClick={() => onCite(citation)}
                  title={`${citation.path}:${citation.start_line}-${citation.end_line}`}
                >
                  {ref}
                </button>
              );
            },
          }}
        >
          {linkCitations(response.answer)}
        </ReactMarkdown>
      </div>

      <ol className={styles.sources}>
        {response.citations.map((c) => (
          <li key={c.ref}>
            <button type="button" className={styles.sourceRow} data-active={isActive(c)} onClick={() => onCite(c)}>
              <span className={`${styles.sourceRef} mono`}>{c.ref}</span>
              <span className={`${styles.sourcePath} mono`} title={c.path}>
                {shortPath(c.path)}
              </span>
              <span className={`${styles.sourceLines} mono`}>
                L{c.start_line}–{c.end_line}
              </span>
            </button>
          </li>
        ))}
      </ol>
      <p className={`${styles.meta} mono`}>
        {response.mode.replace("_", " + ")} · {(response.latency_ms / 1000).toFixed(1)}s
      </p>
    </div>
  );
}
