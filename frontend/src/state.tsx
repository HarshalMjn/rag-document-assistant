import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { api, type ChatResponse, type Config, type DocumentOut, type Health, type Message } from "./api";

/** One answered question with everything the Trace page shows about it. */
export interface TraceEntry {
  key: string;
  conversationId: string;
  question: string;
  response: ChatResponse;
  // The conversation's messages before this question, oldest first.
  priorMessages: Message[];
  askedAt: number;
}

interface AppData {
  documents: DocumentOut[];
  documentsError: string | null;
  refreshDocuments: () => Promise<void>;
  config: Config | null;
  health: Health | null;
  traces: TraceEntry[];
  addTrace: (entry: TraceEntry) => void;
}

const AppDataContext = createContext<AppData | null>(null);

// Traces exist only in the chat response, never in stored history, so they are kept for
// this browser session. Storage can be unavailable (private mode); then they live in memory.
const TRACE_KEY = "docintel.traces";
const MAX_TRACES = 30;

function loadTraces(): TraceEntry[] {
  try {
    const raw = sessionStorage.getItem(TRACE_KEY);
    return raw ? (JSON.parse(raw) as TraceEntry[]) : [];
  } catch {
    return [];
  }
}

function saveTraces(traces: TraceEntry[]) {
  try {
    sessionStorage.setItem(TRACE_KEY, JSON.stringify(traces));
  } catch {
    // Not fatal: traces still work until the page is reloaded.
  }
}

export function AppDataProvider({ children }: { children: ReactNode }) {
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [traces, setTraces] = useState<TraceEntry[]>(loadTraces);

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await api.documents());
      setDocumentsError(null);
    } catch (e) {
      setDocumentsError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
    api.config().then(setConfig, () => setConfig(null));
  }, [refreshDocuments]);

  useEffect(() => {
    let cancelled = false;
    const check = () =>
      api.health().then(
        (h) => !cancelled && setHealth(h),
        // 503 means degraded; any other failure means the API is unreachable.
        (e) =>
          !cancelled &&
          setHealth(
            e?.body && typeof e.body === "object" && "status" in e.body
              ? (e.body as Health)
              : null,
          ),
      );
    check();
    const timer = setInterval(check, 15000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const addTrace = useCallback((entry: TraceEntry) => {
    setTraces((current) => {
      const next = [entry, ...current].slice(0, MAX_TRACES);
      saveTraces(next);
      return next;
    });
  }, []);

  const value = useMemo(
    () => ({ documents, documentsError, refreshDocuments, config, health, traces, addTrace }),
    [documents, documentsError, refreshDocuments, config, health, traces, addTrace],
  );
  return <AppDataContext.Provider value={value}>{children}</AppDataContext.Provider>;
}

export function useAppData(): AppData {
  const value = useContext(AppDataContext);
  if (!value) throw new Error("useAppData must be used inside AppDataProvider");
  return value;
}
