import { useRef, useState } from "react";

import { api, type DocumentOut, type DocumentVersion } from "../api";
import { CloseIcon, InfoIcon, PlusIcon, TrashIcon } from "../components/Icons";
import { StatusPill } from "../components/StatusPill";
import { UploadBanners } from "../components/UploadBanners";
import {
  activeVersion,
  failedUpdate,
  fileSize,
  fileType,
  latestVersion,
  relativeTime,
  shortHash,
  shortId,
  shownVersion,
} from "../format";
import { useAppData } from "../state";
import { useUploader } from "../useUploader";

type ReplaceState = {
  doc: DocumentOut;
  file: File;
  from: DocumentVersion | undefined;
  phase: "running" | "done" | "failed" | "duplicate" | "error";
  result?: DocumentOut;
  error?: string;
};

type DeleteState = {
  doc: DocumentOut;
  phase: "confirm" | "running" | "done" | "error";
  removed?: number;
  error?: string;
};

export function DocumentsPage() {
  const { documents, documentsError, refreshDocuments, config } = useAppData();
  const { upload, pending, results, dismiss } = useUploader();
  const uploadInput = useRef<HTMLInputElement>(null);
  const replaceInput = useRef<HTMLInputElement>(null);
  const [replaceTarget, setReplaceTarget] = useState<DocumentOut | null>(null);
  const [replace, setReplace] = useState<ReplaceState | null>(null);
  const [removal, setRemoval] = useState<DeleteState | null>(null);

  const startReplace = async (doc: DocumentOut, file: File) => {
    const from = activeVersion(doc);
    setReplace({ doc, file, from, phase: "running" });
    try {
      const { document, duplicate } = await api.replace(doc.id, file);
      const latest = latestVersion(document);
      setReplace({
        doc,
        file,
        from,
        result: document,
        phase: duplicate ? "duplicate" : latest?.status === "indexed" ? "done" : "failed",
      });
    } catch (e) {
      setReplace({ doc, file, from, phase: "error", error: (e as Error).message });
    }
    await refreshDocuments();
  };

  const confirmDelete = async () => {
    if (!removal) return;
    setRemoval({ ...removal, phase: "running" });
    try {
      const result = await api.remove(removal.doc.id);
      setRemoval({ doc: removal.doc, phase: "done", removed: result.chunks_removed });
      if (replace?.doc.id === removal.doc.id) setReplace(null);
    } catch (e) {
      setRemoval({ ...removal, phase: "error", error: (e as Error).message });
    }
    await refreshDocuments();
  };

  const busy = replace?.phase === "running" || removal?.phase === "running";
  const highlighted = replace?.doc.id ?? (removal?.phase === "confirm" ? removal.doc.id : undefined);

  return (
    <div className="page">
      <div className="page-head">
        <div style={{ flex: "1 1 auto" }}>
          <h1>Documents</h1>
          <p>Every source in the knowledge base, with its active version. Replace and delete act on a stable document_id.</p>
        </div>
        <button className="btn btn-primary" onClick={() => uploadInput.current?.click()} disabled={pending.length > 0}>
          <PlusIcon />
          {pending.length ? `Indexing ${pending[0]}…` : "Upload documents"}
        </button>
        <input
          ref={uploadInput}
          type="file"
          accept=".pdf,.txt"
          multiple
          className="sr-only"
          onChange={(e) => {
            if (e.target.files) upload(e.target.files);
            e.target.value = "";
          }}
        />
        <input
          ref={replaceInput}
          type="file"
          accept=".pdf,.txt"
          className="sr-only"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file && replaceTarget) startReplace(replaceTarget, file);
            e.target.value = "";
          }}
        />
      </div>

      <UploadBanners results={results} onDismiss={dismiss} />
      {removal?.phase === "done" && (
        <div className="banner ok" role="status">
          <InfoIcon color="#17663A" />
          <span className="text">
            <strong>Deleted:</strong> <span className="mono">{removal.doc.filename}</span> — {removal.removed} chunk
            {removal.removed === 1 ? "" : "s"} removed from Qdrant. It can no longer be retrieved.
          </span>
          <button className="dismiss" aria-label="Dismiss" onClick={() => setRemoval(null)}>
            <CloseIcon />
          </button>
        </div>
      )}

      <div className="split">
        <section className="card card-flush" style={{ flex: "999 1 640px" }}>
          <div className="table-wrap">
            <table className="data" style={{ minWidth: 820 }}>
              <thead>
                <tr>
                  <th>Document</th>
                  <th>document_id</th>
                  <th>Status</th>
                  <th>Version</th>
                  <th>Chunks</th>
                  <th>Hash</th>
                  <th>Updated</th>
                  <th style={{ textAlign: "right" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.length === 0 && (
                  <tr>
                    <td colSpan={8} className="empty">
                      {documentsError ?? "No documents yet. Upload a PDF or TXT file to start."}
                    </td>
                  </tr>
                )}
                {documents.map((d) => {
                  const v = shownVersion(d);
                  const failed = failedUpdate(d);
                  return (
                    <tr key={d.id} className={d.id === highlighted ? "selected" : ""}>
                      <td>
                        <div style={{ fontWeight: 600 }}>{d.filename}</div>
                        <div className="small muted">
                          {[fileType(d.filename), v?.page_count ? `${v.page_count} pages` : null, fileSize(v?.file_size ?? null)]
                            .filter(Boolean)
                            .join(" · ")}
                        </div>
                        {d.status === "failed" && v?.error && <div className="note err">{v.error}</div>}
                        {failed && (
                          <div className="note err">
                            v{failed.version} failed: {failed.error}
                          </div>
                        )}
                      </td>
                      <td className="mono small" style={{ color: "var(--text-2)" }} title={d.id}>
                        doc_{shortId(d.id)}
                      </td>
                      <td>
                        <StatusPill status={d.status} />
                      </td>
                      <td className="mono small">{d.active_version != null ? `v${d.active_version}` : "—"}</td>
                      <td>{v?.chunk_count ?? 0}</td>
                      <td className="mono small" style={{ color: "var(--text-2)" }} title={v?.content_hash}>
                        {v ? shortHash(v.content_hash) : "—"}
                      </td>
                      <td style={{ color: "var(--text-2)", whiteSpace: "nowrap" }}>{relativeTime(d.updated_at)}</td>
                      <td>
                        <div className="actions">
                          <button
                            className="btn btn-sm"
                            disabled={busy}
                            onClick={() => {
                              setReplaceTarget(d);
                              replaceInput.current?.click();
                            }}
                          >
                            Replace
                          </button>
                          <button
                            className="btn btn-icon-danger"
                            aria-label={`Delete ${d.filename}`}
                            disabled={busy}
                            onClick={() => setRemoval({ doc: d, phase: "confirm" })}
                          >
                            <TrashIcon />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <div style={{ flex: "1 1 360px", minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}>
          {replace && <ReplacePanel state={replace} chunkSize={config?.chunk_size} overlap={config?.chunk_overlap} onClose={() => setReplace(null)} />}
          {removal && removal.phase !== "done" && (
            <DeletePanel state={removal} onCancel={() => setRemoval(null)} onConfirm={confirmDelete} />
          )}
          {!replace && (!removal || removal.phase === "done") && (
            <section className="card">
              <h2>How updates stay safe</h2>
              <p className="small" style={{ margin: 0, color: "var(--text-2)", lineHeight: 1.5 }}>
                <strong>Replace</strong> indexes the new file as the next version while the current one stays live.
                Only once every chunk is stored does the document switch versions; then the old chunks are deleted.
                If indexing fails, the old version stays active.
              </p>
              <p className="small" style={{ margin: 0, color: "var(--text-2)", lineHeight: 1.5 }}>
                <strong>Delete</strong> marks the document deleted first, so it stops being retrievable at once, then
                removes its chunks from Qdrant.
              </p>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}

type StepState = "done" | "running" | "failed" | "pending" | "skipped";

function ReplacePanel({
  state,
  chunkSize,
  overlap,
  onClose,
}: {
  state: ReplaceState;
  chunkSize?: number;
  overlap?: number;
  onClose: () => void;
}) {
  const { doc, file, from, phase, result } = state;
  const next = result ? latestVersion(result) : undefined;
  const fromLabel = from ? `v${from.version}` : "none";
  const toLabel = phase === "duplicate" ? fromLabel : next ? `v${next.version}` : "next version";

  // The request is synchronous, so while it runs every step shows as in progress; the
  // response tells which steps finished.
  const run = (ok: boolean): StepState => (phase === "running" ? "running" : ok ? "done" : "failed");
  const indexed = phase === "done";
  const steps: { title: string; detail: string; state: StepState }[] =
    phase === "duplicate"
      ? [
          { title: "Upload replacement", detail: `${file.name} · ${fileSize(file.size)}`, state: "done" },
          {
            title: "Compare content hash",
            detail: `Same bytes as the live ${fromLabel} (${from ? shortHash(from.content_hash) : ""}). Nothing re-indexed.`,
            state: "done",
          },
        ]
      : [
          {
            title: "Upload replacement",
            detail: `${file.name} · ${fileSize(file.size)}${next ? ` · new hash ${shortHash(next.content_hash)}` : ""}`,
            state: phase === "error" ? "failed" : "done",
          },
          {
            title: "Extract, chunk, embed",
            detail:
              indexed && next
                ? `${next.page_count ?? 0} page(s) → ${next.chunk_count ?? 0} chunks (${chunkSize ?? "?"} / ${overlap ?? "?"} overlap)`
                : phase === "failed"
                  ? next?.error ?? "Failed"
                  : "Gemini embeddings, batches of up to 100",
            state: phase === "error" ? "skipped" : run(indexed),
          },
          {
            title: `Store in Qdrant as ${toLabel}`,
            detail: `${fromLabel} stays live while this runs`,
            state: phase === "error" || phase === "failed" ? "skipped" : run(indexed),
          },
          {
            title: "Swap active version",
            detail: `documents.active_version ${fromLabel} → ${toLabel}`,
            state: phase === "error" || phase === "failed" ? "skipped" : run(indexed),
          },
          {
            title: `Retire ${fromLabel} chunks`,
            detail: from ? `Deleted points where document_id=doc_${shortId(doc.id)} AND version=${from.version}` : "No previous version",
            state: phase === "error" || phase === "failed" ? "skipped" : run(indexed),
          },
        ];

  return (
    <section className="card panel-replace" aria-live="polite">
      <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
        <div style={{ flex: "1 1 auto" }}>
          <div className="eyebrow">{phase === "running" ? "Replacing" : "Replace"}</div>
          <h2 style={{ margin: "4px 0 2px", fontSize: 16 }}>{doc.filename}</h2>
          <div className="small muted mono">
            doc_{shortId(doc.id)} · {fromLabel} → {toLabel}
          </div>
        </div>
        {phase !== "running" && (
          <button className="dismiss" aria-label="Close" onClick={onClose}>
            <CloseIcon />
          </button>
        )}
      </div>
      <div className="steps">
        {steps.map((s, i) => (
          <div className="step" key={s.title}>
            <div className="step-rail">
              <span className={`step-dot ${s.state}`}>
                {s.state === "done" ? "✓" : s.state === "failed" ? "✕" : s.state === "running" ? "…" : i + 1}
              </span>
              {i < steps.length - 1 && <span className="step-line" />}
            </div>
            <div className="step-body">
              <div className="step-title">{s.title}</div>
              <div className="step-detail">{s.detail}</div>
            </div>
          </div>
        ))}
      </div>
      {phase === "failed" && (
        <div className="hint">
          Indexing failed, so the update is marked <strong style={{ color: "var(--err-fg)" }}>failed</strong>.{" "}
          {fromLabel} stays active and its chunks are untouched; partial chunks of the new version were removed.
        </div>
      )}
      {phase === "error" && <div className="hint" style={{ color: "var(--err-fg)" }}>{state.error}</div>}
      {phase === "running" && (
        <div className="hint">
          If any step before the swap fails, {fromLabel} stays active and the update is marked{" "}
          <strong style={{ color: "var(--err-fg)" }}>failed</strong>. Staged chunks are cleaned up.
        </div>
      )}
      {phase === "done" && (
        <div className="hint">Future answers use {toLabel} only; ask the same question in Chat to check.</div>
      )}
    </section>
  );
}

function DeletePanel({
  state,
  onCancel,
  onConfirm,
}: {
  state: DeleteState;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const v = activeVersion(state.doc);
  return (
    <section className="card panel-delete">
      <h2>Delete {state.doc.filename}?</h2>
      <p className="small" style={{ margin: 0, color: "var(--text-2)", lineHeight: 1.5 }}>
        All {v?.chunk_count ?? 0} chunks for <span className="mono">doc_{shortId(state.doc.id)}</span> are removed from
        Qdrant and the record is marked deleted. Answers already in chat history keep their old citations.
      </p>
      {state.phase === "error" && <div className="note err">{state.error}</div>}
      <div className="actions">
        <button className="btn" onClick={onCancel} disabled={state.phase === "running"}>
          Cancel
        </button>
        <button className="btn btn-danger" onClick={onConfirm} disabled={state.phase === "running"}>
          {state.phase === "running" ? "Deleting…" : "Delete document"}
        </button>
      </div>
    </section>
  );
}
