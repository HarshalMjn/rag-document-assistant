import { useRef, useState, type DragEvent } from "react";
import { Link } from "react-router-dom";

import { failedUpdate, latestVersion, relativeTime, replacedVersion, shownVersion } from "../format";
import { useAppData } from "../state";
import { useUploader } from "../useUploader";
import { UploadIcon } from "./Icons";
import { StatusPill } from "./StatusPill";
import { UploadBanners } from "./UploadBanners";

/** Right-hand panel of the chat workspace: upload plus a compact document list. */
export function KnowledgeBase() {
  const { documents, documentsError, config } = useAppData();
  const { upload, pending, results, dismiss } = useUploader();
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    upload(e.dataTransfer.files);
  };

  return (
    <aside className="kb" aria-label="Knowledge base">
      <div className="kb-head">
        <h2>Knowledge base</h2>
        <Link to="/documents" className="small" style={{ fontWeight: 500 }}>
          Manage
        </Link>
      </div>

      <label
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <UploadIcon />
        <span className="dropzone-title">
          {pending.length ? `Indexing ${pending[0]}…` : "Drop PDF or TXT, or browse"}
        </span>
        <span className="small muted">Max {config?.max_upload_mb ?? 10} MB per file</span>
        <input
          ref={input}
          type="file"
          accept=".pdf,.txt"
          multiple
          className="sr-only"
          disabled={pending.length > 0}
          onChange={(e) => {
            if (e.target.files) upload(e.target.files);
            e.target.value = "";
          }}
        />
      </label>

      <UploadBanners results={results.slice(0, 3)} onDismiss={dismiss} />

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {documentsError && <div className="note err">{documentsError}</div>}
        {!documentsError && documents.length === 0 && (
          <div className="empty">No documents yet. Upload one to start asking questions.</div>
        )}
        {documents.map((d) => {
          const shown = shownVersion(d);
          const latest = latestVersion(d);
          const failed = failedUpdate(d);
          const replaced = replacedVersion(d);
          return (
            <div className="doc-card" key={d.id}>
              <div className="doc-card-top">
                <span className="doc-card-name" title={d.filename}>
                  {d.filename}
                </span>
                <StatusPill status={d.status} />
              </div>
              <span className="small muted mono">
                v{shown?.version ?? "?"}
                {shown?.page_count ? ` · ${shown.page_count} pages` : ""} · {shown?.chunk_count ?? 0} chunks
              </span>
              {d.status === "failed" && latest?.error && <span className="note err">{latest.error}</span>}
              {failed && (
                <span className="note err">
                  Update to v{failed.version} failed; v{d.active_version} is still live.
                </span>
              )}
              {!failed && replaced && (
                <span className="note">
                  Updated {relativeTime(d.updated_at)} — v{replaced.version} chunks retired
                </span>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
}
