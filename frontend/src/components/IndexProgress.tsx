import type { RepoStatus } from "@/lib/api";
import styles from "./Workbench.module.css";

const STAGES = ["cloning", "chunking", "embedding", "ready"] as const;
const LABELS: Record<(typeof STAGES)[number], string> = {
  cloning: "clone",
  chunking: "chunk",
  embedding: "embed",
  ready: "ready",
};

export function IndexProgress({ job, onRetry }: { job: RepoStatus; onRetry: () => void }) {
  const current = job.status === "queued" ? -1 : STAGES.indexOf(job.status as (typeof STAGES)[number]);
  const failed = job.status === "failed";

  return (
    <section className={styles.progress} aria-live="polite">
      <div className={styles.progressHead}>
        <span className="mono">{job.repo}</span>
        {job.commit && <span className={`${styles.commit} mono`}>@{job.commit.slice(0, 7)}</span>}
        <span className={styles.progressMsg}>
          {failed ? "Indexing failed" : job.status === "ready" ? `${job.chunks} chunks from ${job.files_indexed} files` : job.message}
        </span>
        {job.status === "ready" && (
          <button type="button" className={`${styles.linkButton} mono`} onClick={onRetry}>
            re-index
          </button>
        )}
      </div>

      <ol className={styles.stages}>
        {STAGES.map((stage, i) => (
          <li
            key={stage}
            className={`${styles.stage} mono`}
            data-state={failed && i >= Math.max(current, 0) ? "failed" : i < current || job.status === "ready" ? "done" : i === current ? "active" : "todo"}
          >
            {LABELS[stage]}
          </li>
        ))}
      </ol>

      <div className={styles.bar} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(job.progress * 100)}>
        <div className={styles.barFill} data-failed={failed} style={{ width: `${Math.max(4, job.progress * 100)}%` }} />
      </div>

      {failed && (
        <p className={styles.error}>
          {job.error}{" "}
          <button type="button" className={`${styles.linkButton} mono`} onClick={onRetry}>
            try again
          </button>
        </p>
      )}
    </section>
  );
}
