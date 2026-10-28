import type { DocumentStatus } from "../api";
import { statusLabel } from "../format";

export function StatusPill({ status }: { status: DocumentStatus }) {
  return <span className={`pill pill-${status}`}>{statusLabel(status)}</span>;
}
