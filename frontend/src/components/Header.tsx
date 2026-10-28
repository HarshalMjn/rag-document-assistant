import { NavLink } from "react-router-dom";

import { activeVersion } from "../format";
import { useAppData } from "../state";
import { LogoIcon } from "./Icons";

export function Header() {
  const { documents, config, health } = useAppData();
  const indexed = documents.filter((d) => d.active_version != null);
  const chunks = indexed.reduce((sum, d) => sum + (activeVersion(d)?.chunk_count ?? 0), 0);

  const healthy = health?.status === "ok";
  const healthText = !health
    ? "API unreachable"
    : healthy
      ? "API healthy"
      : `Degraded · qdrant ${health.qdrant} · mysql ${health.mysql}`;

  return (
    <header className="header">
      <div className="brand">
        <LogoIcon />
        <span className="brand-name">DocIntel</span>
        <span className="brand-sub">Document Intelligence Assistant</span>
      </div>
      <div className="spacer" />
      <div className="chips">
        <span className={`chip chip-health ${healthy ? "ok" : "bad"}`} role="status">
          <span className="dot" />
          {healthText}
        </span>
        <span className="chip">
          qdrant · {indexed.length} source{indexed.length === 1 ? "" : "s"} indexed · {chunks} chunks
        </span>
        {config && (
          <span className="chip" title={`LLM ${config.llm_model} · rewrite ${config.rewrite_model}`}>
            chunk {config.chunk_size} / overlap {config.chunk_overlap} · top-k {config.top_k}
          </span>
        )}
      </div>
      <nav className="nav" aria-label="Main">
        <NavLink to="/chat">Chat</NavLink>
        <NavLink to="/documents">Documents</NavLink>
        <NavLink to="/trace">Trace</NavLink>
      </nav>
    </header>
  );
}
