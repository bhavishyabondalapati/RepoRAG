import type { Citation } from "@/lib/api";
import styles from "./Workbench.module.css";

export function SourcePanel({ citation }: { citation: Citation | null }) {
  if (!citation) {
    return (
      <div className={styles.sourceEmpty}>
        <p className={`${styles.sourceEmptyTitle} mono`}>source</p>
        <p>Click a citation to see the exact lines an answer came from.</p>
      </div>
    );
  }

  const lines = citation.snippet.split("\n");
  return (
    <div className={styles.viewer}>
      <div className={styles.viewerHead}>
        <div className={styles.viewerTitle}>
          <span className={`${styles.viewerPath} mono`}>{citation.path}</span>
          <span className={`${styles.viewerMeta} mono`}>
            L{citation.start_line}–{citation.end_line}
            {citation.symbol ? ` · ${citation.symbol}` : ""} · {citation.language}
          </span>
        </div>
        <a className={`${styles.ghLink} mono`} href={citation.url} target="_blank" rel="noreferrer">
          GitHub ↗
        </a>
      </div>
      <pre className={styles.code}>
        <code>
          {lines.map((line, i) => (
            <span key={i} className={styles.codeLine}>
              <span className={styles.lineNo} aria-hidden>
                {citation.start_line + i}
              </span>
              <span className={styles.lineText}>{line || " "}</span>
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}
