import type { DocumentOut, DocumentVersion, Message } from "./api";

/** MySQL returns naive timestamps in UTC; mark them as UTC before parsing. */
export function parseDate(value: string): Date {
  return new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(value) ? value : `${value}Z`);
}

export function relativeTime(value: string, now = Date.now()): string {
  const date = parseDate(value);
  const seconds = Math.round((now - date.getTime()) / 1000);
  if (seconds < 45) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  if (hours < 48) return "yesterday";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function shortId(id: number): string {
  return String(id);
}

export function shortHash(hash: string): string {
  return `${hash.slice(0, 4)}…${hash.slice(-4)}`;
}

export function fileSize(bytes: number | null): string {
  if (bytes == null) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function activeVersion(doc: DocumentOut): DocumentVersion | undefined {
  return doc.versions.find((v) => v.version === doc.active_version);
}

export function latestVersion(doc: DocumentOut): DocumentVersion | undefined {
  return doc.versions[doc.versions.length - 1];
}

/** The version a row should describe: the live one, or the newest when none is live yet. */
export function shownVersion(doc: DocumentOut): DocumentVersion | undefined {
  return activeVersion(doc) ?? latestVersion(doc);
}

export function fileType(filename: string): string {
  const ext = filename.split(".").pop()?.toUpperCase();
  return ext && ext !== filename.toUpperCase() ? ext : "FILE";
}

export function statusLabel(status: string): string {
  return status.charAt(0).toUpperCase() + status.slice(1);
}

/** A newer version failed while an older one is still live. */
export function failedUpdate(doc: DocumentOut): DocumentVersion | undefined {
  const latest = latestVersion(doc);
  return latest && latest.status === "failed" && doc.active_version != null
    ? latest
    : undefined;
}

/** The version an update replaced: the newest indexed one below the active version. */
export function replacedVersion(doc: DocumentOut): DocumentVersion | undefined {
  return doc.versions
    .filter((v) => doc.active_version != null && v.version < doc.active_version && v.status === "indexed")
    .pop();
}

/**
 * How many of `messages` the backend sends as context for the next question: the latest
 * `limit`, minus a leading assistant reply whose question was cut off. Mirrors
 * ChatService.recent_history.
 */
export function contextWindow(messages: Message[], limit: number): number {
  if (limit <= 0) return 0;
  const window = messages.slice(-limit);
  let start = 0;
  while (start < window.length && window[start].role === "assistant") start++;
  return window.length - start;
}

export function sourceLabel(filename: string, page: number | null): string {
  return page != null ? `${filename} · p.${page}` : filename;
}
