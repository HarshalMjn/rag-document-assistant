"""Retrieval and answer evaluation over the sample documents (assignment section 7).

Indexes samples/ once per chunking config, each into its own throwaway Qdrant collection,
then measures Recall@K, MRR and refusals, and checks real answers for the default config.
Uses the real Gemini embeddings and Groq LLM through the app's own services, but skips
MySQL: every file is treated as live version 1 of its own document, so dev data is never
touched.

    docker compose exec fastapi_app python -m eval.run_eval
    docker compose exec fastapi_app python -m eval.run_eval --no-answers   # no Groq calls

Results go to eval/results/: a timestamped JSON per run, plus latest.json and latest.md.
Token counts are estimates (characters / 4), not billed usage.
"""

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from app.clients import vector_store_client
from app.clients.embedding_client import EmbeddingClient
from app.clients.llm_client import LLMClientError
from app.clients.vector_store_client import SearchHit
from app.config.settings import settings
from app.schemas.chunk import ChunkPayload
from app.services.answer_service import AnswerService
from app.services.chunking_service import ChunkingService
from app.services.extraction_service import ExtractionService
from app.services.prompt_service import INSUFFICIENT_ANSWER, PromptService

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
QUESTIONS_FILE = EVAL_DIR / "questions.json"
_ROOT = EVAL_DIR.parent
SAMPLES = (
    _ROOT / "samples" if (_ROOT / "samples").is_dir() else _ROOT.parent / "samples"
)

# (chunk_size, chunk_overlap). The first one is the app default and gets the answer check.
CONFIGS = [
    (settings.CHUNK_SIZE, settings.CHUNK_OVERLAP),
    (400, 60),
    (1200, 200),
]
K_VALUES = (1, 3, 5)
CHARS_PER_TOKEN = 4
# Gemini's free tier allows 100 embedded texts per minute (each text in a batch counts).
EMBED_TEXTS_PER_MINUTE = 90


class EmbedRateLimit:
    """Keeps embedding under the free-tier limit by waiting before a call would exceed it.

    Time spent waiting is added to `waited`, so latency numbers can leave it out.
    """

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.sent: list[tuple[float, int]] = []
        self.waited = 0.0

    async def acquire(self, count: int) -> None:
        while True:
            now = time.monotonic()
            self.sent = [(t, n) for t, n in self.sent if now - t < 60]
            if sum(n for _, n in self.sent) + count <= self.per_minute:
                self.sent.append((now, count))
                return
            pause = 60 - (now - self.sent[0][0]) + 0.1
            print(f"  waiting {pause:.0f}s for the embedding rate limit...")
            await asyncio.sleep(pause)
            self.waited += pause


limiter = EmbedRateLimit(EMBED_TEXTS_PER_MINUTE)


def throttle_embeddings() -> None:
    """Route every embedding call through the rate limit, in this process only."""
    embed = EmbeddingClient._embed

    async def throttled(texts: list[str], task_type: str) -> list[list[float]]:
        await limiter.acquire(len(texts))
        return await embed(texts, task_type)

    EmbeddingClient._embed = staticmethod(throttled)
    settings.EMBEDDING_BATCH_SIZE = min(
        settings.EMBEDDING_BATCH_SIZE, EMBED_TEXTS_PER_MINUTE
    )


def elapsed(started: float, waited_before: float) -> float:
    """Seconds since `started`, minus any rate-limit waiting in between."""
    return time.perf_counter() - started - (limiter.waited - waited_before)


# Models often write non-breaking or en dashes ("5‑10", "5–10") where the source has "-".
_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-"})


def normalize(text: str) -> str:
    """Lower-case, dashes unified and all whitespace collapsed (including PDF line breaks
    and the narrow no-break spaces models write, as in "31 March")."""
    return re.sub(r"\s+", " ", text.translate(_DASHES)).strip().lower()


def is_hit(chunk: ChunkPayload, question: dict) -> bool:
    """The chunk comes from the expected file and page and contains the evidence phrase."""
    return (
        question["source"] == chunk.filename
        and (question["pages"] is None or chunk.page in question["pages"])
        and normalize(question["evidence"]) in normalize(chunk.text)
    )


def has_fact(answer: str, fact: str) -> bool:
    return any(normalize(option) in normalize(answer) for option in fact.split("|"))


def tokens(chars: int) -> int:
    return round(chars / CHARS_PER_TOKEN)


def collection_name(size: int, overlap: int) -> str:
    return f"eval_{size}_{overlap}"


# Indexing


async def index_config(size: int, overlap: int, documents: list[str]) -> dict:
    """Chunk, embed and store every document into this config's fresh collection."""
    vector_store_client.qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
    vector_store_client.ensure_collection()

    started, waited = time.perf_counter(), limiter.waited
    chunk_count = chunk_chars = 0
    for document_id, filename in enumerate(documents, start=1):
        data = (SAMPLES / filename).read_bytes()
        pages = ExtractionService.extract(data, filename)
        chunks = ChunkingService.chunk_pages(
            pages, chunk_size=size, chunk_overlap=overlap
        )
        payloads = [
            ChunkPayload(
                document_id=document_id,
                version=1,
                chunk_id=vector_store_client.make_chunk_id(
                    document_id, 1, chunk.chunk_index
                ),
                chunk_index=chunk.chunk_index,
                filename=filename,
                page=chunk.page,
                content_hash=hashlib.sha256(data).hexdigest(),
                text=chunk.text,
            )
            for chunk in chunks
        ]
        vectors = await EmbeddingClient.embed_documents([p.text for p in payloads])
        vector_store_client.upsert_chunks(payloads, vectors)
        chunk_count += len(payloads)
        chunk_chars += sum(len(p.text) for p in payloads)

    return {
        "chunk_count": chunk_count,
        "index_seconds": round(elapsed(started, waited), 2),
        "index_tokens_estimate": tokens(chunk_chars),
    }


# Retrieval


async def retrieve(question: str, active_versions: list[tuple[int, int]]):
    """Same search as RetrievalService, minus the MySQL lookup of live versions."""
    started, waited = time.perf_counter(), limiter.waited
    vector = await EmbeddingClient.embed_query(question)
    hits = vector_store_client.search(
        vector, active_versions, max(max(K_VALUES), settings.TOP_K)
    )
    seconds = elapsed(started, waited)
    evidence = [h for h in hits[: settings.TOP_K] if h.score >= settings.MIN_SCORE]
    return hits, evidence, seconds


def retrieval_result(question: dict, hits: list[SearchHit], evidence, seconds) -> dict:
    answerable = question["source"] is not None
    rank = (
        next(
            (i for i, h in enumerate(hits, start=1) if is_hit(h.payload, question)),
            None,
        )
        if answerable
        else None
    )
    return {
        "id": question["id"],
        "answerable": answerable,
        "rank": rank,
        "top_score": round(hits[0].score, 4) if hits else None,
        "evidence_chunks": len(evidence),
        "retrieval_seconds": round(seconds, 3),
        "top_chunks": [
            {
                "filename": h.payload.filename,
                "page": h.payload.page,
                "score": round(h.score, 4),
                "hit": answerable and is_hit(h.payload, question),
                "text": h.payload.text[:200],
            }
            for h in hits[:3]
        ],
    }


def summarize_retrieval(results: list[dict]) -> dict:
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    ranks = [r["rank"] for r in answerable]
    answerable_scores = [
        r["top_score"] for r in answerable if r["top_score"] is not None
    ]
    unanswerable_scores = [
        r["top_score"] for r in unanswerable if r["top_score"] is not None
    ]
    return {
        **{
            f"recall@{k}": round(
                sum(rank is not None and rank <= k for rank in ranks) / len(ranks), 3
            )
            for k in K_VALUES
        },
        "mrr": round(sum(1 / rank for rank in ranks if rank) / len(ranks), 3),
        # Refused at retrieval: no chunk reached MIN_SCORE, so the LLM is never called.
        "refused_at_retrieval": sum(r["evidence_chunks"] == 0 for r in unanswerable),
        "unanswerable_total": len(unanswerable),
        "answerable_top_score": _spread(answerable_scores),
        "unanswerable_top_score": _spread(unanswerable_scores),
        "mean_retrieval_seconds": round(
            statistics.mean(r["retrieval_seconds"] for r in results), 3
        ),
    }


def _spread(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "min": min(values),
        "median": round(statistics.median(values), 4),
        "max": max(values),
    }


# Answers


async def answer(question: dict, evidence: list[SearchHit]) -> dict:
    """Generate the real answer the way AnswerService does for a first question."""
    started = time.perf_counter()
    error = None
    if evidence:
        try:
            text, used = await AnswerService._generate(
                question["question"], evidence, []
            )
        except LLMClientError as e:
            text, used, error = "", [], str(e)
    else:
        text, used = INSUFFICIENT_ANSWER, []
    seconds = time.perf_counter() - started

    prompt = PromptService.build(question["question"], evidence)
    prompt_chars = len(prompt.system_instruction) + sum(
        len(m["content"]) for m in prompt.messages
    )
    refused = text == INSUFFICIENT_ANSWER
    if question["source"] is None:
        correct = refused
        facts_ok = citation_ok = None
    else:
        facts_ok = all(has_fact(text, fact) for fact in question["answer_facts"])
        citation_ok = any(is_hit(hit.payload, question) for hit in used)
        correct = facts_ok and citation_ok

    return {
        "answer": text,
        "error": error,
        "refused": refused,
        "facts_ok": facts_ok,
        "citation_ok": citation_ok,
        "correct": correct,
        "citations": [
            {"filename": hit.payload.filename, "page": hit.payload.page} for hit in used
        ],
        "generation_seconds": round(seconds, 3),
        "llm_called": bool(evidence),
        "prompt_tokens_estimate": tokens(prompt_chars) if evidence else 0,
        "answer_tokens_estimate": tokens(len(text)) if evidence else 0,
    }


def summarize_answers(results: list[dict]) -> dict:
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    called = [r for r in results if r["llm_called"]]
    return {
        "facts_ok": sum(bool(r["facts_ok"]) for r in answerable),
        "citation_ok": sum(bool(r["citation_ok"]) for r in answerable),
        "answerable_correct": sum(bool(r["correct"]) for r in answerable),
        "answerable_total": len(answerable),
        "refused_correctly": sum(r["refused"] for r in unanswerable),
        "unanswerable_total": len(unanswerable),
        "errors": sum(r["error"] is not None for r in results),
        "llm_calls": len(called),
        "mean_generation_seconds": round(
            statistics.mean(r["generation_seconds"] for r in called), 3
        )
        if called
        else None,
        "prompt_tokens_estimate": sum(r["prompt_tokens_estimate"] for r in results),
        "answer_tokens_estimate": sum(r["answer_tokens_estimate"] for r in results),
    }


# Run


async def run(with_answers: bool) -> dict:
    spec = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    documents, questions = spec["documents"], spec["questions"]
    active_versions = [(i, 1) for i in range(1, len(documents) + 1)]
    original_collection = settings.QDRANT_COLLECTION
    configs = []

    try:
        for size, overlap in CONFIGS:
            name = collection_name(size, overlap)
            settings.QDRANT_COLLECTION = name
            print(f"[{name}] indexing {len(documents)} documents...")
            indexing = await index_config(size, overlap, documents)

            results = []
            for question in questions:
                hits, evidence, seconds = await retrieve(
                    question["question"], active_versions
                )
                result = retrieval_result(question, hits, evidence, seconds)
                if with_answers and (size, overlap) == CONFIGS[0]:
                    result |= await answer(question, evidence)
                results.append(result)
            print(f"[{name}] {len(results)} questions done")

            config = {
                "chunk_size": size,
                "chunk_overlap": overlap,
                "default": (size, overlap) == CONFIGS[0],
                **indexing,
                "retrieval": summarize_retrieval(results),
                "questions": results,
            }
            if with_answers and config["default"]:
                config["answers"] = summarize_answers(results)
            configs.append(config)
    finally:
        for size, overlap in CONFIGS:
            vector_store_client.qdrant_client.delete_collection(
                collection_name(size, overlap)
            )
        settings.QDRANT_COLLECTION = original_collection

    return {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "answer_model": settings.CHAT_ANSWER_MODEL,
            "top_k": settings.TOP_K,
            "min_score": settings.MIN_SCORE,
            "embedding_batch_size": settings.EMBEDDING_BATCH_SIZE,
        },
        "documents": documents,
        "question_count": len(questions),
        "answerable_count": sum(q["source"] is not None for q in questions),
        "configs": configs,
    }


# Report


def to_markdown(report: dict) -> str:
    s = report["settings"]
    lines = [
        "# Evaluation results",
        "",
        f"Run at {report['run_at']} with `{s['embedding_model']}` embeddings and "
        f"`{s['answer_model']}` answers; TOP_K={s['top_k']}, MIN_SCORE={s['min_score']}. "
        f"{report['question_count']} questions ({report['answerable_count']} answerable) "
        f"over {len(report['documents'])} documents. Token counts are estimates (chars / 4).",
        "",
        "## Retrieval per chunking config",
        "",
        "| Config | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR | Refused at retrieval "
        "| Answerable top score (min / median) | Unanswerable top score (max) "
        "| Index s | Index tokens (est.) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report["configs"]:
        r = c["retrieval"]
        a, u = r["answerable_top_score"], r["unanswerable_top_score"]
        lines.append(
            f"| {c['chunk_size']}/{c['chunk_overlap']}{' (default)' if c['default'] else ''} "
            f"| {c['chunk_count']} | {r['recall@1']} | {r['recall@3']} | {r['recall@5']} "
            f"| {r['mrr']} | {r['refused_at_retrieval']}/{r['unanswerable_total']} "
            f"| {a['min']} / {a['median']} | {u['max'] if u else '-'} "
            f"| {c['index_seconds']} | {c['index_tokens_estimate']} |"
        )

    lines += ["", "## Rank of the evidence chunk per question", ""]
    header = " | ".join(
        f"{c['chunk_size']}/{c['chunk_overlap']}" for c in report["configs"]
    )
    lines += [f"| Question | {header} |", "|---|" + "---|" * len(report["configs"])]
    for i, question in enumerate(report["configs"][0]["questions"]):
        ranks = []
        for c in report["configs"]:
            q = c["questions"][i]
            if q["answerable"]:
                ranks.append(str(q["rank"] or "miss"))
            else:
                ranks.append("refused" if q["evidence_chunks"] == 0 else "not refused")
        lines.append(f"| {question['id']} | {' | '.join(ranks)} |")

    default = report["configs"][0]
    if "answers" in default:
        a = default["answers"]
        lines += [
            "",
            f"## Answers ({default['chunk_size']}/{default['chunk_overlap']})",
            "",
            f"Facts correct {a['facts_ok']}/{a['answerable_total']}, citations correct "
            f"{a['citation_ok']}/{a['answerable_total']}, fully correct "
            f"{a['answerable_correct']}/{a['answerable_total']}, refused correctly "
            f"{a['refused_correctly']}/{a['unanswerable_total']}, errors {a['errors']}. "
            f"{a['llm_calls']} LLM calls, mean {a['mean_generation_seconds']} s; about "
            f"{a['prompt_tokens_estimate']} prompt + {a['answer_tokens_estimate']} answer "
            "tokens (est.).",
            "",
            "| Question | Correct | Facts | Citation | Cited | Answer |",
            "|---|---|---|---|---|---|",
        ]
        for q in default["questions"]:
            cited = ", ".join(
                f"{c['filename']}" + (f" p{c['page']}" if c["page"] else "")
                for c in q["citations"]
            )
            text = (q["error"] or q["answer"]).replace("\n", " ").replace("|", "\\|")
            lines.append(
                f"| {q['id']} | {_mark(q['correct'])} | {_mark(q['facts_ok'])} "
                f"| {_mark(q['citation_ok'])} | {cited or '-'} | {text[:160]} |"
            )
    return "\n".join(lines) + "\n"


def _mark(value: bool | None) -> str:
    return "-" if value is None else ("yes" if value else "NO")


def save(report: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.fromisoformat(report["run_at"]).strftime("%Y-%m-%d_%H%M%S")
    body = json.dumps(report, indent=2, ensure_ascii=False)
    run_file = RESULTS_DIR / f"{stamp}.json"
    run_file.write_text(body, encoding="utf-8")
    (RESULTS_DIR / "latest.json").write_text(body, encoding="utf-8")
    (RESULTS_DIR / "latest.md").write_text(to_markdown(report), encoding="utf-8")
    return run_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-answers",
        action="store_true",
        help="only measure retrieval; skips the Groq answer check",
    )
    args = parser.parse_args()

    throttle_embeddings()
    report = asyncio.run(run(with_answers=not args.no_answers))
    run_file = save(report)
    print()
    print(to_markdown(report))
    print(f"Saved {run_file.relative_to(EVAL_DIR.parent)} and latest.json / latest.md")


if __name__ == "__main__":
    main()
