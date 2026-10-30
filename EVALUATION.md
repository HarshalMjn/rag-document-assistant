# Evaluation report

Retrieval and answer quality of the assistant on a small, self-written HR knowledge base,
for two chunking configurations. Every number below comes from one run of
`backend/eval/run_eval.py`; the raw per-question data is in `backend/eval/results/results.json` and the
generated tables in `backend/eval/results/summary.md`.

## Setup

| | |
|---|---|
| Corpus | `samples/`: `employee_handbook.pdf` (3 pages: leave, remote work, travel), `it_security_policy.txt`, `benefits_guide.txt`. About 4,200 characters in total. |
| Question set | `backend/eval/questions.json`: 22 questions. 19 answerable, including 2 that need evidence from two documents (q18, q19) and 2 follow-ups that only make sense after the previous question (q05, q16). 3 unanswerable, including a near-miss trap (q20 "parental leave bonus": the corpus has parental leave and a referral bonus, but no parental leave bonus). |
| Models | Embeddings `gemini-embedding-001` (768 dims). Answers `openai/gpt-oss-120b`, follow-up rewrites `openai/gpt-oss-20b`, both on Groq. |
| Retrieval | Cosine similarity in Qdrant, top-k = 5, `MIN_SCORE` = 0.55 |
| Config A | 800-character chunks, 120 overlap (the shipped default): 7 chunks |
| Config B | 400-character chunks, 60 overlap: 15 chunks |

Each configuration is indexed into its own Qdrant collection, and the pipeline is the
production code (`ExtractionService`, `ChunkingService`, `EmbeddingClient`,
`vector_store_client.search`, `AnswerService` rewrite and generation). Follow-ups are asked
with the previous question and the answer the run just produced as history.

### Metric definitions

- **Recall@K**: for each answerable question, the share of its evidence passages found in
  a top-K chunk, averaged over questions. A passage is "found" when a chunk from the right
  file contains its text (case and whitespace ignored). For q18/q19, finding 1 of 2
  passages scores 0.5.
- **MRR**: mean of 1 / rank of the first chunk holding evidence.
- **Evidence reaches the LLM**: every evidence chunk is in the top 5 and scores at least
  `MIN_SCORE`, so it is actually sent to the LLM.
- **Answer correct**: answerable questions are answered and contain every expected
  keyword; unanswerable ones get the "not enough information" answer with no citations.
  This is checked automatically, then every answer was read by hand (below).
- **Citation correct**: at least one cited chunk holds the evidence. **Citation
  precision**: the share of cited chunks that hold evidence.

## Results

| Metric | A (800/120) | B (400/60) |
|---|---|---|
| Chunks indexed | 7 | 15 |
| **Recall@1** | **0.895** | 0.789 |
| **Recall@3** | **1.000** | 0.947 |
| **Recall@5** | 1.000 | 1.000 |
| **MRR** | **0.974** | 0.912 |
| Evidence reaches the LLM | 1.000 | 1.000 |
| Evidence chunk score: mean / lowest | 0.662 / 0.565 | 0.694 / 0.593 |
| Unanswerable: highest top score | 0.660 | 0.686 |
| Answer correct, automatic check | 20/22 | 20/22 |
| Answer correct, manual review | **22/22** | **22/22** |
| Unanswerable correctly refused | 3/3 | 3/3 |
| Citation correct / precision | 1.000 / 1.000 | 1.000 / 0.974 |
| LLM input tokens per answer (est.) | ~1,060 | ~750 |

### Qualitative review

- **Answers.** All 22 answers are correct in both configurations. The 2 automatic
  failures are false negatives in the keyword check, not wrong answers: q05 says "two
  consecutive days" (keyword "2") and q09 says "six hours" (keyword "6 hours").
- **Citations.** Every answer cites a chunk that contains its evidence, and nothing is
  invented. The multi-source questions q18 and q19 cite both documents in both
  configurations. The one imprecise citation is B/q02, which also cites the neighbouring
  leave chunk.
- **Follow-ups.** Both were rewritten correctly before retrieval: "Do I need a certificate
  for those?" became "Do I need a certificate for the paid sick leave days?" and "And
  the secondary caregiver?" became "How many weeks of parental leave does the secondary
  caregiver get?". The evidence was ranked first both times.
- **Unanswerable.** All 3 were refused with no citation. But the refusal came from the
  LLM, not from the score threshold (see below).

## Chunking: what changed between A and B

- **Smaller chunks score higher but rank worse.** In B each chunk covers fewer topics, so
  the matching chunk is more similar to the question (mean evidence score 0.694 vs 0.662).
  But facts are split across more chunks with similar wording, so the right one lands
  first less often: Recall@1 drops from 0.895 to 0.789, and MRR from 0.974 to 0.912.
  For example, for q07 (home office allowance) the evidence is rank 3 in B and rank 1 in A,
  and for q18 the home-office chunk is rank 4 in B and rank 2 in A.
- **With top-k = 5 both reach Recall@5 = 1.0**, so on this corpus the answers end up the same.
- **B sends about 29% fewer tokens to the LLM** (750 vs 1,060 per answer), because 5 small
  chunks are shorter than 5 large ones.
- **Decision: keep 800/120.** It ranks evidence first more often, needs fewer embeddings,
  and with ~200-token chunks the prompt stays small. B would matter on a larger corpus
  where top-k cannot cover a whole document. This corpus fits in 7 chunks, so it cannot
  show that; the comparison should be repeated on more data.

## The relevance threshold cannot separate unanswerable questions

The lowest-scoring evidence chunk scored 0.565 in A, while unanswerable questions reached
0.660 in A and 0.686 in B (q20 matches the parental leave chunk well). The two ranges
overlap, so no `MIN_SCORE` keeps every correct chunk and drops every unrelated one. This
is why `MIN_SCORE` is a low floor (0.55) that only drops clearly unrelated chunks, and the
LLM's `answerable` flag makes the final decision.

## Failure cases and what to improve next

| Failure | Seen in | Improvement |
|---|---|---|
| Unanswerable questions reach the LLM with high-scoring but irrelevant chunks; refusal depends entirely on the model | q20 (0.66 to 0.69), q22 | Add a cross-encoder reranker or hybrid BM25 + vector search, whose scores separate better than raw cosine similarity. Keep the `answerable` flag as the last line of defence. |
| Small chunks split a section, so the evidence drops in rank | B/q07, B/q18 | Keep 800/120; on larger corpora try section-aware chunking (split on headings), or attach the section heading to each chunk. |
| Ambiguous follow-ups can be resolved to the wrong topic | Manual testing: "Can those be carried over?" after a sick leave answer was rewritten to "paid leave days" | Return `search_query` (done) so it is visible. Give the rewrite model the previous answer's citations, or ask a clarifying question when the reference is ambiguous. |
| Keyword scoring misses number words ("two", "six") | q05, q09 | Normalise numbers before comparing, or use an LLM judge for answer correctness. |
| Latency spikes from free-tier rate limits | A/q14: 50 s (generation waited for Groq's rate-limit retries) | A paid tier, or a fallback model when the limit is hit. |
| Small, self-written test set (22 questions, 3 short documents) gives optimistic numbers | — | Add longer documents, tables, multi-page sections, contradictory versions, and questions written by someone other than the document author. |

## Latency and cost

### Latency (measured in this run, Docker on a laptop, free-tier APIs)

| Step | Median | Notes |
|---|---|---|
| Ingestion: extract + chunk | < 0.1 s per document | PyMuPDF / plain text |
| Ingestion: embed | 0.5 to 1.3 s per document | One Gemini batch call (up to 100 chunks) per document |
| Ingestion: store in Qdrant | < 0.1 s per document | Waits until Qdrant has applied the write |
| Question: rewrite (follow-ups only) | 1.6 to 2.4 s | Only 2 samples; first questions skip it |
| Question: embed + Qdrant search | ~0.95 s | Almost all of it is the Gemini embedding call |
| Question: LLM answer | 1.5 s (A), 3.2 s (B) | p90 5 to 10 s; maximum 50 s when the free tier rate-limited us |
| **Question end to end** | **3.2 s (A), 4.5 s (B)** | Includes a rewrite only for follow-ups |

The A/B difference in answer time is provider variance, not chunk size: B sends fewer
tokens, yet was slower.

### Cost (estimates)

Token counts are estimated as characters / 4. The prices are assumptions based on list
prices at the time of writing: about $0.15 per 1M input tokens for Gemini embeddings and,
for `gpt-oss-120b` on Groq, about $0.15 per 1M input and $0.60 per 1M output tokens.
Check the providers' current pricing before relying on these numbers.

| Item | Tokens (est.) | Cost (est.) |
|---|---|---|
| Embedding the whole sample corpus | ~1,060 | ~$0.0002 |
| Ingesting a 100-page document (~250k characters) | ~62,000 | ~$0.01 |
| One question: query embedding | ~15 | negligible |
| One question: answer prompt | ~1,060 in (A) | ~$0.00016 |
| One question: answer output | ~20 visible, plus reasoning tokens (not measured, assume ~300) | ~$0.0002 |
| **1,000 questions** | | **~$0.35 to $0.50** |

During development everything runs on free tiers at $0. Those tiers limit requests per
day and tokens per minute, which is what caused the latency spikes above.

## Reproduce

```bash
docker compose up -d
docker compose exec fastapi_app python -m eval.run_eval
```

The run takes a few minutes and uses its own `eval_c800` / `eval_c400` collections, so the
real knowledge base is untouched. Because real models are used, scores and timings vary
slightly between runs.
