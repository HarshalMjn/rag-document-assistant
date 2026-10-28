import { useCallback, useState } from "react";

import { api, type DocumentOut } from "./api";
import { latestVersion } from "./format";
import { useAppData } from "./state";

export interface UploadResult {
  id: number;
  filename: string;
  kind: "indexed" | "duplicate" | "failed" | "error";
  message: string;
  document?: DocumentOut;
}

let nextId = 1;

/** Uploads files one at a time (each request returns only once indexing is done). */
export function useUploader() {
  const { refreshDocuments } = useAppData();
  const [pending, setPending] = useState<string[]>([]);
  const [results, setResults] = useState<UploadResult[]>([]);

  const upload = useCallback(
    async (files: FileList | File[]) => {
      const list = Array.from(files);
      if (!list.length) return;
      setPending(list.map((f) => f.name));
      for (const file of list) {
        const result = await uploadOne(file);
        setResults((current) => [result, ...current]);
        setPending((current) => current.filter((name) => name !== file.name));
        await refreshDocuments();
      }
    },
    [refreshDocuments],
  );

  const dismiss = useCallback(
    (id: number) => setResults((current) => current.filter((r) => r.id !== id)),
    [],
  );

  return { upload, pending, results, dismiss };
}

async function uploadOne(file: File): Promise<UploadResult> {
  const id = nextId++;
  try {
    const { document, duplicate } = await api.upload(file);
    if (duplicate) {
      return {
        id,
        filename: file.name,
        kind: "duplicate",
        document,
        message: `has the same content hash as ${document.filename}. Nothing was re-indexed.`,
      };
    }
    const version = latestVersion(document);
    if (version?.status === "failed") {
      return {
        id,
        filename: file.name,
        kind: "failed",
        document,
        message: version.error ?? "Indexing failed.",
      };
    }
    return {
      id,
      filename: file.name,
      kind: "indexed",
      document,
      message: `indexed: ${version?.page_count ?? 0} page(s) → ${version?.chunk_count ?? 0} chunks.`,
    };
  } catch (e) {
    return { id, filename: file.name, kind: "error", message: (e as Error).message };
  }
}
