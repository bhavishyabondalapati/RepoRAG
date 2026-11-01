# RepoRAG frontend

Next.js + TypeScript UI for RepoRAG. See the [root README](../README.md) for setup.

```bash
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

- `src/components/Workbench.tsx`: page state (repo input, indexing poll, chat turns)
- `src/components/AnswerBody.tsx`: renders answers and turns `[n]` markers into clickable citations
- `src/components/SourcePanel.tsx`: shows the cited lines with real line numbers
- `src/components/IndexProgress.tsx`: clone → chunk → embed → ready progress
- `src/lib/api.ts`: typed client for the FastAPI backend
