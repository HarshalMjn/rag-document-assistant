"""Evaluate retrieval and answers over samples/ for two chunking configurations.

Run inside the API container, which has the API keys and dependencies:

    docker compose exec fastapi_app python -m eval.run_eval

Each configuration is indexed into its own Qdrant collection (eval_c<size>), so the real
knowledge base is never touched, and MySQL is not used. Results are written to
eval/results/results.json (every question) and eval/results/summary.md (the tables).
Embeddings and answers use the real providers (Gemini, Groq), so numbers vary slightly
between runs.
"""

import asyncio
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from app.config.settings import settings
from app.schemas.chunk import ChunkPayload
from app.services import qdrant_service
from app.services.chat_service import ChatService
from app.services.chunking_service import ChunkingService
from app.services.extraction_service import ExtractionService
from app.services.gemini_service import GeminiService
from app.services.llm_service import ChatMessage, LLMServiceError
from app.services.prompt_service import INSUFFICIENT_ANSWER, PromptService

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
QUESTIONS = Path(__file__).parent / "questions.json"
RESULTS = Path(__file__).parent / "results"

# (name, chunk size, overlap). A is the shipped default; B halves both.
CONFIGS = [("A", 800, 120), ("B", 400, 60)]
K_VALUES = (1, 3, 5)
TOP_K = max(K_VALUES)
SAMPLE_FILES = ["employee_handbook.pdf", "it_security_policy.txt", "benefits_guide.txt"]

# Groq's free tier limits tokens per minute; on a 429 wait and try again.
_LLM_RETRIES = 4
_LLM_RETRY_WAIT_S = 30


def norm(text: str) -> str:
    """Lowercase, single spaces, no thousands separators: PDF text breaks lines mid-sentence."""
    return re.sub(r"\s+", " ", text.lower()).replace(",", "").strip()


def tokens(chars: int) -> int:
    """Rough token estimate (about 4 characters per token for English)."""
    return round(chars / 4)


@dataclass
class Indexed:
    document_ids: dict[str, int]
    chunks: int
    chars: int
    timings: dict[str, dict[str, int]]


async def index(config: str, size: int, overlap: int) -> Indexed:
    settings.QDRANT_COLLECTION = f"eval_c{size}"
    qdrant_service.qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
    qdrant_service.ensure_collection()

    document_ids, timings = {}, {}
    total_chunks = total_chars = 0
    for document_id, filename in enumerate(SAMPLE_FILES, start=1):
        document_ids[filename] = document_id
        data = (SAMPLES / filename).read_bytes()

        start = time.perf_counter()
        pages = ExtractionService.extract(data, filename)
        chunks = ChunkingService.chunk_pages(
            pages, chunk_size=size, chunk_overlap=overlap
        )
        chunked = time.perf_counter()
        payloads = [
            ChunkPayload(
                document_id=document_id,
                version=1,
                chunk_id=qdrant_service.make_chunk_id(document_id, 1, c.chunk_index),
                chunk_index=c.chunk_index,
                filename=filename,
                page=c.page,
                content_hash="eval",
                text=c.text,
            )
            for c in chunks
        ]
        vectors = await GeminiService.embed_documents([p.text for p in payloads])
        embedded = time.perf_counter()
        qdrant_service.upsert_chunks(payloads, vectors)
        stored = time.perf_counter()

        total_chunks += len(payloads)
        total_chars += sum(len(p.text) for p in payloads)
        timings[filename] = {
            "chunks": len(payloads),
            "extract_chunk_ms": ms(start, chunked),
            "embed_ms": ms(chunked, embedded),
            "store_ms": ms(embedded, stored),
        }
        print(f"[{config}] indexed {filename}: {len(payloads)} chunks")
    return Indexed(document_ids, total_chunks, total_chars, timings)


def ms(start: float, end: float) -> int:
    return round((end - start) * 1000)


async def with_llm_retry(make_call):
    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            return await make_call()
        except LLMServiceError as e:
            if "(429)" not in str(e) or attempt == _LLM_RETRIES:
                raise
            print(f"  rate limited, waiting {_LLM_RETRY_WAIT_S}s")
            await asyncio.sleep(_LLM_RETRY_WAIT_S)


def evidence_ranks(hits, evidence) -> list[int | None]:
    """1-based rank of the first chunk holding each evidence item, or None."""
    ranks = []
    for item in evidence:
        needle = norm(item["text"])
        rank = next(
            (
                i
                for i, hit in enumerate(hits, start=1)
                if hit.payload.filename == item["file"]
                and needle in norm(hit.payload.text)
            ),
            None,
        )
        ranks.append(rank)
    return ranks


async def ask(
    q: dict, history: list[ChatMessage], active: list[tuple[str, int]]
) -> dict:
    started = time.perf_counter()
    query = q["question"]
    if history:
        query = await with_llm_retry(
            lambda: ChatService._standalone_query(q["question"], history)
        )
    rewritten = time.perf_counter()

    vector = await GeminiService.embed_query(query)
    hits = qdrant_service.search(vector, active, TOP_K)
    retrieved = time.perf_counter()

    evidence = [h for h in hits if h.score >= settings.MIN_SCORE]
    prompt_chars = 0
    if evidence:
        prompt = PromptService.build(q["question"], evidence, history)
        prompt_chars = len(prompt.system_instruction) + sum(
            len(m["content"]) for m in prompt.messages
        )
        answer, used = await with_llm_retry(
            lambda: ChatService._generate(q["question"], evidence, history)
        )
    else:
        answer, used = INSUFFICIENT_ANSWER, []
    generated = time.perf_counter()

    ranks = evidence_ranks(hits, q["evidence"])
    record = {
        "id": q["id"],
        "question": q["question"],
        "search_query": query,
        "answerable": bool(q["evidence"]),
        "hits": [
            {
                "file": h.payload.filename,
                "page": h.payload.page,
                "chunk_index": h.payload.chunk_index,
                "score": round(h.score, 4),
                "in_prompt": h.score >= settings.MIN_SCORE,
            }
            for h in hits
        ],
        "evidence_ranks": ranks,
        "answer": answer,
        "insufficient": not used,
        "cited": [
            {"file": u.payload.filename, "chunk_index": u.payload.chunk_index}
            for u in used
        ],
        "timings_ms": {
            "rewrite": ms(started, rewritten),
            "retrieval": ms(rewritten, retrieved),
            "generation": ms(retrieved, generated),
            "total": ms(started, generated),
        },
        "llm_input_tokens_est": tokens(prompt_chars),
        "llm_output_tokens_est": tokens(len(answer)) if evidence else 0,
    }
    record.update(judge(q, record, hits, used))
    return record


def judge(q: dict, record: dict, hits, used) -> dict:
    """Automatic checks; summary.md lists every answer for a manual second look."""
    if not q["evidence"]:
        return {
            "answer_correct": record["insufficient"] and not used,
            "citation_correct": not used,
        }

    text = norm(record["answer"])
    answer_correct = not record["insufficient"] and all(
        norm(k) in text for k in q["answer_keywords"]
    )
    needles = [(e["file"], norm(e["text"])) for e in q["evidence"]]
    cited_with_evidence = [
        u
        for u in used
        if any(
            u.payload.filename == f and n in norm(u.payload.text) for f, n in needles
        )
    ]
    return {
        "answer_correct": answer_correct,
        # At least one citation points at a chunk that holds the evidence.
        "citation_correct": bool(cited_with_evidence),
        "citation_precision": round(len(cited_with_evidence) / len(used), 3)
        if used
        else 0.0,
    }


def summarize(
    name: str, size: int, overlap: int, indexed: Indexed, records: list[dict]
) -> dict:
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    def recall_at(k: int) -> float:
        # Share of evidence items found in the top k, averaged over questions.
        per_question = [
            sum(1 for rank in r["evidence_ranks"] if rank is not None and rank <= k)
            / len(r["evidence_ranks"])
            for r in answerable
        ]
        return round(statistics.mean(per_question), 3)

    def first_rank(r):
        found = [rank for rank in r["evidence_ranks"] if rank is not None]
        return min(found) if found else None

    mrr = statistics.mean(1 / first_rank(r) if first_rank(r) else 0 for r in answerable)
    evidence_scores = [
        r["hits"][rank - 1]["score"]
        for r in answerable
        for rank in r["evidence_ranks"]
        if rank is not None
    ]
    reached_llm = [
        all(
            rank is not None and r["hits"][rank - 1]["in_prompt"]
            for rank in r["evidence_ranks"]
        )
        for r in answerable
    ]
    top_unanswerable = [r["hits"][0]["score"] for r in unanswerable if r["hits"]]
    answered = [r for r in answerable if not r["insufficient"]]

    def stat(step: str) -> dict:
        values = sorted(r["timings_ms"][step] for r in records)
        return {"mean": round(statistics.mean(values)), "max": values[-1]}

    return {
        "config": name,
        "chunk_size": size,
        "chunk_overlap": overlap,
        "chunks": indexed.chunks,
        "questions": len(records),
        "answerable": len(answerable),
        "recall_at_k": {str(k): recall_at(k) for k in K_VALUES},
        "mrr": round(mrr, 3),
        "evidence_reaches_llm": round(sum(reached_llm) / len(answerable), 3),
        "evidence_score_mean": round(statistics.mean(evidence_scores), 3),
        "evidence_score_min": round(min(evidence_scores), 3),
        "unanswerable_top_score_max": round(max(top_unanswerable), 3)
        if top_unanswerable
        else None,
        "answer_accuracy": round(
            sum(r["answer_correct"] for r in records) / len(records), 3
        ),
        "answerable_accuracy": round(
            sum(r["answer_correct"] for r in answerable) / len(answerable), 3
        ),
        "unanswerable_correct": f"{sum(r['answer_correct'] for r in unanswerable)}/{len(unanswerable)}",
        "citation_correct": round(
            sum(r["citation_correct"] for r in answered) / len(answered), 3
        )
        if answered
        else 0,
        "citation_precision": round(
            statistics.mean(r["citation_precision"] for r in answered), 3
        )
        if answered
        else 0,
        "latency_ms": {
            step: stat(step) for step in ("rewrite", "retrieval", "generation", "total")
        },
        "ingestion": {
            "chars": indexed.chars,
            "embedding_tokens_est": tokens(indexed.chars),
            "per_file": indexed.timings,
        },
        "llm_input_tokens_est_mean": round(
            statistics.mean(
                r["llm_input_tokens_est"] for r in records if r["llm_input_tokens_est"]
            )
        ),
        "llm_output_tokens_est_mean": round(
            statistics.mean(
                r["llm_output_tokens_est"]
                for r in records
                if r["llm_output_tokens_est"]
            )
        ),
    }


def markdown(summaries: list[dict], runs: dict[str, list[dict]]) -> str:
    a, b = summaries
    rows = [
        ("Chunks indexed", "chunks"),
        ("Recall@1", ("recall_at_k", "1")),
        ("Recall@3", ("recall_at_k", "3")),
        ("Recall@5", ("recall_at_k", "5")),
        ("MRR", "mrr"),
        (
            f"Evidence reaches the LLM (score >= {settings.MIN_SCORE})",
            "evidence_reaches_llm",
        ),
        ("Evidence chunk score, mean", "evidence_score_mean"),
        ("Evidence chunk score, lowest", "evidence_score_min"),
        ("Unanswerable: highest top score", "unanswerable_top_score_max"),
        ("Answer accuracy, all questions", "answer_accuracy"),
        ("Answer accuracy, answerable", "answerable_accuracy"),
        ("Unanswerable correctly refused", "unanswerable_correct"),
        ("Citation correct (answered questions)", "citation_correct"),
        ("Citation precision", "citation_precision"),
        ("Mean total latency (ms)", ("latency_ms", "total", "mean")),
        ("Mean LLM generation latency (ms)", ("latency_ms", "generation", "mean")),
        ("Mean retrieval latency (ms)", ("latency_ms", "retrieval", "mean")),
        ("LLM input tokens per answer (est.)", "llm_input_tokens_est_mean"),
    ]

    def get(summary, key):
        if isinstance(key, tuple):
            value = summary
            for part in key:
                value = value[part]
            return value
        return summary[key]

    lines = [
        f"| Metric | A ({a['chunk_size']}/{a['chunk_overlap']}) | B ({b['chunk_size']}/{b['chunk_overlap']}) |",
        "|---|---|---|",
    ]
    lines += [f"| {label} | {get(a, key)} | {get(b, key)} |" for label, key in rows]
    for name, records in runs.items():
        lines += [
            "",
            f"### Config {name}: every question",
            "",
            "| id | evidence rank | top score | ok | cited | answer |",
            "|---|---|---|---|---|---|",
        ]
        for r in records:
            ranks = (
                ",".join("-" if x is None else str(x) for x in r["evidence_ranks"])
                or "n/a"
            )
            top = r["hits"][0]["score"] if r["hits"] else "-"
            cited = (
                ", ".join(f"{c['file']}#{c['chunk_index']}" for c in r["cited"]) or "-"
            )
            answer = r["answer"].replace("|", "/").replace("\n", " ")
            lines.append(
                f"| {r['id']} | {ranks} | {top} | {'yes' if r['answer_correct'] else 'NO'} | {cited} | {answer} |"
            )
    return "\n".join(lines) + "\n"


async def main():
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    by_id = {q["id"]: q for q in questions}
    RESULTS.mkdir(exist_ok=True)
    original_collection = settings.QDRANT_COLLECTION

    summaries, runs = [], {}
    try:
        for name, size, overlap in CONFIGS:
            indexed = await index(name, size, overlap)
            active = [(document_id, 1) for document_id in indexed.document_ids.values()]
            records, answers = [], {}
            for q in questions:
                history: list[ChatMessage] = []
                if parent := q.get("follow_up_of"):
                    history = [
                        {"role": "user", "content": by_id[parent]["question"]},
                        {"role": "assistant", "content": answers[parent]},
                    ]
                record = await ask(q, history, active)
                answers[q["id"]] = record["answer"]
                records.append(record)
                print(
                    f"[{name}] {q['id']} ranks={record['evidence_ranks']} ok={record['answer_correct']}"
                )
            runs[name] = records
            summaries.append(summarize(name, size, overlap, indexed, records))
    finally:
        settings.QDRANT_COLLECTION = original_collection

    (RESULTS / "results.json").write_text(
        json.dumps(
            {
                "min_score": settings.MIN_SCORE,
                "llm_model": settings.CHAT_ANSWER_MODEL,
                "rewrite_model": settings.CHAT_REWRITE_MODEL,
                "embedding_model": settings.EMBEDDING_MODEL,
                "summaries": summaries,
                "runs": runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (RESULTS / "summary.md").write_text(markdown(summaries, runs), encoding="utf-8")
    print((RESULTS / "summary.md").read_text(encoding="utf-8").split("\n### ")[0])


if __name__ == "__main__":
    asyncio.run(main())
