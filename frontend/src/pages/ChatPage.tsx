import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { api, type ConversationSummary, type Message } from "../api";
import { AnswerCard } from "../components/AnswerCard";
import { HistoryIcon, PlusIcon, SendIcon } from "../components/Icons";
import { KnowledgeBase } from "../components/KnowledgeBase";
import { contextWindow, relativeTime, shortId } from "../format";
import { useAppData } from "../state";

export function ChatPage() {
  const params = useParams();
  const conversationId = params.conversationId ? Number(params.conversationId) : undefined;
  const navigate = useNavigate();
  const { config, traces, addTrace } = useAppData();
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState<string | null>(null);
  const [askError, setAskError] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const refreshConversations = useCallback(async () => {
    try {
      const list = await api.conversations();
      setConversations(list);
      return list;
    } catch (e) {
      setLoadError((e as Error).message);
      return [];
    }
  }, []);

  // Open the most recent conversation when none is selected.
  useEffect(() => {
    refreshConversations().then((list) => {
      if (!conversationId && list.length) navigate(`/chat/${list[0].id}`, { replace: true });
    });
  }, [conversationId, navigate, refreshConversations]);

  useEffect(() => {
    setMessages([]);
    setAskError(null);
    if (!conversationId) return;
    api.history(conversationId).then(
      (h) => {
        setMessages(h.messages);
        setLoadError(null);
      },
      (e) => setLoadError((e as Error).message),
    );
  }, [conversationId]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, asking]);

  const newConversation = async () => {
    try {
      const created = await api.createConversation();
      await refreshConversations();
      navigate(`/chat/${created.id}`);
    } catch (e) {
      setLoadError((e as Error).message);
    }
  };

  const send = async (e: FormEvent) => {
    e.preventDefault();
    const text = question.trim();
    if (!text || asking) return;
    let id = conversationId;
    if (!id) {
      id = (await api.createConversation()).id;
      navigate(`/chat/${id}`, { replace: true });
    }
    const prior = messages;
    setAsking(text);
    setAskError(null);
    setQuestion("");
    try {
      const response = await api.ask(id, text);
      const { messages: updated } = await api.history(id);
      const answer = updated[updated.length - 1];
      addTrace({
        key: `${id}:${answer.id}`,
        conversationId: id,
        question: text,
        response,
        priorMessages: prior,
        askedAt: Date.now(),
      });
      setMessages(updated);
      refreshConversations();
    } catch (err) {
      // Nothing was stored on the server, so give the question back for a retry.
      setAskError((err as Error).message);
      setQuestion(text);
    } finally {
      setAsking(null);
    }
  };

  const current = conversations.find((c) => c.id === conversationId);
  const limit = config?.history_limit ?? 10;
  const used = contextWindow(messages, limit);
  const traceFor = (m: Message) => traces.find((t) => t.key === `${conversationId}:${m.id}`);

  return (
    <div className="workspace">
      <aside className="sidebar" aria-label="Conversations">
        <button className="btn btn-primary" onClick={newConversation}>
          <PlusIcon />
          New conversation
        </button>
        <div className="section-label">Conversations</div>
        {conversations.length === 0 && <div className="empty">No conversations yet.</div>}
        {conversations.map((c) => (
          <button
            key={c.id}
            className={`convo ${c.id === conversationId ? "active" : ""}`}
            onClick={() => navigate(`/chat/${c.id}`)}
          >
            <span className="convo-title">{c.title ?? "New conversation"}</span>
            <span className="convo-meta">
              {c.message_count} messages · {relativeTime(c.last_activity)}
            </span>
          </button>
        ))}
      </aside>

      <main className="chat-main">
        <div className="chat-bar">
          <h1>{current?.title ?? (conversationId ? "New conversation" : "Start a conversation")}</h1>
          {conversationId && <span className="small muted mono">conv_{shortId(conversationId)}</span>}
          <div className="spacer" />
          {conversationId && (
            <span
              className="pill pill-blue"
              style={{ display: "inline-flex", alignItems: "center", gap: 6, fontWeight: 500 }}
              title="Messages sent as context with the next question (user + assistant combined)"
            >
              <HistoryIcon />
              Context: last {used} of {messages.length} messages
            </span>
          )}
        </div>

        <div className="thread" aria-live="polite">
          {loadError && <div className="answer error">{loadError}</div>}
          {!loadError && messages.length === 0 && !asking && (
            <div className="empty">
              Ask a question about your documents. Follow-ups can refer to earlier answers; the
              latest {limit} messages are sent as context.
            </div>
          )}
          {messages.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="bubble-user">
                {m.content}
              </div>
            ) : (
              <AnswerCard key={m.id} message={m} trace={traceFor(m)} minScore={config?.min_score} />
            ),
          )}
          {asking && (
            <>
              <div className="bubble-user">{asking}</div>
              <div className="answer-row">
                <div className="avatar">AI</div>
                <div className="answer">
                  <span className="typing">Rewriting, retrieving and answering…</span>
                </div>
              </div>
            </>
          )}
          {askError && (
            <div className="answer-row">
              <div className="avatar">AI</div>
              <div className="answer error">
                <strong>The question could not be answered.</strong>
                <span>{askError}</span>
                <span className="small">Nothing was saved; your question is back in the box to retry.</span>
              </div>
            </div>
          )}
          <div ref={bottom} />
        </div>

        <form className="composer" onSubmit={send}>
          <div className="composer-inner">
            <label>
              <span className="small muted">Ask about your documents</span>
              <input
                className="input"
                type="text"
                value={question}
                maxLength={4000}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g. Can unused leave be carried over?"
                disabled={!!asking}
              />
            </label>
            <button type="submit" className="send" aria-label="Send question" disabled={!!asking || !question.trim()}>
              <SendIcon />
            </button>
          </div>
        </form>
      </main>

      <KnowledgeBase />
    </div>
  );
}
