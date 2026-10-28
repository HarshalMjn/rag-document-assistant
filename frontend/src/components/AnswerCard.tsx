import { Link } from "react-router-dom";

import type { Message, TraceItem } from "../api";
import { shortId } from "../format";
import type { TraceEntry } from "../state";
import { AlertIcon } from "./Icons";

interface Props {
  message: Message;
  trace?: TraceEntry;
  minScore?: number;
}

export function AnswerCard({ message, trace, minScore }: Props) {
  const response = trace?.response;
  // Stored answers keep [] citations when the knowledge base had no answer.
  const insufficient = response ? response.insufficient_evidence : (message.citations ?? []).length === 0;
  const byChunk = new Map<string, TraceItem>(
    (response?.retrieval_trace ?? []).map((t) => [t.chunk_id, t]),
  );
  const rewritten =
    response && trace && response.search_query.trim() !== trace.question.trim()
      ? response.search_query
      : null;

  if (insufficient) {
    return (
      <div className="answer-row">
        <div className="avatar">AI</div>
        <div className="answer insufficient">
          <AlertIcon />
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <strong>Insufficient information</strong>
            <span className="body">
              {message.content} {response && insufficientReason(response.retrieval_trace, minScore)} No
              citations returned.
            </span>
            {rewritten && <span className="interpreted">Searched as: “{rewritten}”</span>}
            {trace && (
              <Link to={`/trace/${encodeURIComponent(trace.key)}`} className="small">
                Open retrieval trace
              </Link>
            )}
          </div>
        </div>
      </div>
    );
  }

  const citations = message.citations ?? [];
  return (
    <div className="answer-row">
      <div className="avatar">AI</div>
      <div className="answer">
        {rewritten && (
          <div className="interpreted">
            Interpreted as: <em>“{rewritten}”</em>
          </div>
        )}
        <p>{message.content}</p>
        {citations.length > 0 && (
          <div className="citations">
            {citations.map((c, i) => {
              const t = byChunk.get(c.chunk_id);
              const parts = [
                c.filename,
                c.page != null ? `p.${c.page}` : null,
                t ? `chunk ${t.chunk_index}` : null,
                t ? `v${t.version}` : null,
              ].filter(Boolean);
              return (
                <span className="citation" key={c.chunk_id} title={`chunk_id ${c.chunk_id}`}>
                  <strong>[{i + 1}]</strong>
                  {parts.join(" · ")}
                </span>
              );
            })}
          </div>
        )}
        {response && trace && (
          <details className="trace-details">
            <summary>
              Retrieval trace · {response.retrieval_trace.length} chunks
              {response.timings ? ` · ${response.timings.total_ms} ms` : ""}
            </summary>
            <div className="trace-lines">
              {response.retrieval_trace.map((t) => (
                <div key={t.chunk_id} className={t.in_prompt ? "" : "dim"}>
                  {shortId(t.document_id)}:v{t.version}:c{t.chunk_index}&nbsp;&nbsp;{t.score.toFixed(3)}
                  &nbsp;&nbsp;{t.used_in_answer ? "cited" : t.in_prompt ? "in prompt" : "below threshold"}
                </div>
              ))}
            </div>
            <Link to={`/trace/${encodeURIComponent(trace.key)}`} style={{ display: "inline-block", marginTop: 8 }}>
              Open full trace
            </Link>
          </details>
        )}
      </div>
    </div>
  );
}

function insufficientReason(trace: TraceItem[], minScore?: number): string {
  if (trace.length === 0) return "No document chunk was retrieved.";
  const best = Math.max(...trace.map((t) => t.score));
  if (!trace.some((t) => t.in_prompt)) {
    return `Best match scored ${best.toFixed(2)}, below the ${minScore ?? "configured"} relevance threshold.`;
  }
  return `The best ${trace.filter((t) => t.in_prompt).length} chunk(s) (top score ${best.toFixed(2)}) did not contain the answer.`;
}
