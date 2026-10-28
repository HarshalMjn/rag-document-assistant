import type { UploadResult } from "../useUploader";
import { CloseIcon, InfoIcon } from "./Icons";

const TITLES: Record<UploadResult["kind"], string> = {
  indexed: "Uploaded:",
  duplicate: "Duplicate skipped:",
  failed: "Indexing failed:",
  error: "Upload rejected:",
};

export function UploadBanners({
  results,
  onDismiss,
}: {
  results: UploadResult[];
  onDismiss: (id: number) => void;
}) {
  return (
    <>
      {results.map((r) => (
        <div
          key={r.id}
          role="status"
          className={`banner ${r.kind === "indexed" ? "ok" : r.kind === "duplicate" ? "" : "err"}`}
        >
          <InfoIcon color={r.kind === "indexed" ? "#17663A" : r.kind === "duplicate" ? "#1C3BA8" : "#A3201A"} />
          <span className="text">
            <strong>{TITLES[r.kind]}</strong> <span className="mono">{r.filename}</span> {r.message}
          </span>
          <button className="dismiss" aria-label="Dismiss" onClick={() => onDismiss(r.id)}>
            <CloseIcon />
          </button>
        </div>
      ))}
    </>
  );
}
