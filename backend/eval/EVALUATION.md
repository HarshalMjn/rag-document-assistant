# Evaluation report

How well the RAG pipeline finds the right evidence and answers from it, measured on the
sample documents with the real models. Every number below comes from
[`results/2026-10-30_093325.json`](results/2026-10-30_093325.json) (same content as
[`results/latest.json`](results/latest.json); readable tables in
[`results/latest.md`](results/latest.md)).

## 1. Setup

| | |
|---|---|
| Documents | `leave_policy_v2.txt` (fictional HR policy), `diabetes_how_to_take_control.pdf` (6 pages), `eatwell_guide_2018.pdf` (12 pages) |
| Questions | 21 in [`questions.json`](questions.json): 17 answerable (6 leave, 6 diabetes, 5 Eatwell) and 4 that no document answers |
| Embeddings | `gemini-embedding-001`, 768 dimensions, cosine similarity in Qdrant |
| Answer model | `openai/gpt-oss-120b` on Groq, temperature 0 |
| Retrieval | `TOP_K = 5`, `MIN_SCORE = 0.55` (app defaults) |
| Run | 30 October 2026, one run, free tiers of both providers |

Most questions are worded differently from the document ("doctor's note" for *medical
certificate*, "work from home" for *remotely*, "main carer of a newborn" for *primary
caregiver*), so they test semantic search rather than keyword overlap. The no-answer
questions include two on the same topic as a document (bereavement leave, daily iron
intake), which are the hardest to refuse.

Reproduce:

```
docker compose exec fastapi_app python -m eval.run_eval
```

The script ([`run_eval.py`](run_eval.py)) calls the app's own extraction, chunking,
embedding, Qdrant and answer code. It skips MySQL and FastAPI, indexes each chunking config
into its own throwaway Qdrant collection, and never touches dev data.

## 2. Retrieval metric

- **Hit:** a retrieved chunk that comes from the expected file (and page, for PDFs) and
  contains the question's `evidence` phrase, compared case-insensitively with whitespace
  and dashes normalised.
- **Recall@K:** share of the 17 answerable questions with a hit among the top K chunks.
- **MRR** (mean reciprocal rank): average of 1 / rank of the first hit (0 when there is none).
  1.0 means the evidence chunk was always first.
- **Refused at retrieval:** no-answer questions where no chunk reached `MIN_SCORE`, so the
  LLM was never called.

## 3. Chunking configurations

| Config (size / overlap, chars) | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR | Mean retrieval |
|---|---|---|---|---|---|---|
| **800 / 120 (default)** | 66 | 0.706 | **0.941** | **1.000** | 0.819 | 0.73 s |
| 400 / 60 | 126 | 0.588 | 0.765 | 0.765 | 0.667 | 0.53 s |
| 1200 / 200 | 47 | **0.765** | 0.882 | **1.000** | **0.850** | 0.66 s |

What changed:

- **400 / 60 missed 4 of 17 questions completely** (leave-02, diabetes-05, eatwell-01,
  eatwell-02). In every case the top chunk was from the right passage, but the fact was
  **split across a chunk boundary**. For leave-02, the top chunk begins *"over into the next
  year; carried-over days must be used by 31 March"*, so it has the deadline but not
  *"Up to 5 unused days"*. The strict hit rule counts this as a miss, and the LLM would only
  have gotten half the fact. 60 characters of overlap is less than one line of these
  documents, so it rarely saves a split sentence.
- **1200 / 200 ranks slightly better at the top** (Recall@1 0.765, MRR 0.85), because a
  big chunk usually contains the whole passage. But big chunks dilute short facts:
  *"Aim for ½ of your plate to be vegetables"* (diabetes-03) fell from rank 1 to rank 4,
  behind three general Eatwell chunks (pages 3, 10 and 2) about food groups.
- **800 / 120 is the only config with the evidence in the top 3 for 16 of 17 questions and
  in the top 5 for all 17.** With `TOP_K = 5`, every fact reached the prompt.

**Decision: keep 800 / 120.** It has the best Recall@3 and full Recall@5. 1200 / 200 is
close, but it puts more text into every prompt and missed the top 3 for a short fact. The
difference between 800 and 1200 is two questions, which is too few to treat as a clear win.

## 4. Answer and citation review (800 / 120)

| | Result |
|---|---|
| Facts correct (answer contains every expected fact) | **17 / 17** |
| Citation correct (a cited chunk holds the evidence, right file and page) | **17 / 17** |
| No-answer questions refused | **4 / 4** |
| LLM or provider errors | 0 |

Examples (full answers in `results/latest.md`):

- **leave-03**, *"When do I need a doctor's note?"*: "You need a medical certificate
  (doctor's note) if you are absent for more than three consecutive working days." Correct
  despite the different wording. Cited `leave_policy_v2.txt`.
- **diabetes-06**, *"How much exercise per week?"*: "150 minutes of moderate-intensity
  activity each week (or 75 minutes of vigorous activity), plus muscle-strengthening
  exercises twice a week." Complete, and cited page 5.
- **none-01**, *"What is the bereavement leave policy?"*: the fixed insufficient-evidence
  reply, with no citation, even though five leave-policy chunks were sent as context
  (see section 5).
- **diabetes-04**, *"Calories in a pint of beer or a glass of wine?"*: this answer is
  correct but weak (see section 6).

A note on scoring: the first run marked two correct answers as wrong. The model wrote
`31 March` (a narrow no-break space) and "three" instead of "3". The fact check now
normalises whitespace and dashes the same way as the evidence check, and leave-03 accepts
`3|three`. The run above is the rerun after that fix; no answer was changed by hand.

## 5. Refusals and `MIN_SCORE`

| Top similarity score (800 / 120) | Min | Median | Max |
|---|---|---|---|
| Answerable questions | 0.631 | 0.718 | 0.797 |
| No-answer questions | 0.526 | 0.597 | **0.647** |

**The threshold does not separate the two groups.** The two ranges overlap: bereavement
leave (0.647) and daily iron intake (0.618) score higher than the weakest answerable
question (0.631). So retrieval refused only **1 of 4**, the off-topic *"capital of France"*
(0.526). The other three passed the threshold, and the **LLM refused them**, following the
"answer only from the sources" rule. The same pattern holds in all three configs.

This is the intended design: `MIN_SCORE` is a cheap first filter that skips the LLM call
for clearly unrelated questions, and the grounded prompt is the real safeguard. Raising the
threshold to about 0.65 would block the three on-topic questions, but it would also drop
answerable questions scoring 0.63–0.65. Lowering it would cost an extra LLM call for
off-topic questions and change nothing else.

## 6. Failure cases and what I would improve

1. **Facts split across chunk boundaries** (all four 400 / 60 misses). Chunks are cut by
   character count with a small overlap, so a sentence can straddle two chunks.
   *Next:* make the overlap at least one full sentence (sentence-aware overlap), or attach
   the neighbouring chunk when a hit lands at a chunk edge.
2. **Weak ranking for diabetes-04** (rank 4 at 800, rank 5 at 1200). Eatwell's alcohol
   paragraph (*"1 pint of standard strength lager contains approximately 136kcal"*) is
   closer to the question than the diabetes sentence with the expected *"170-250
   calories"*. Both are true. The answer reported both, but it read *"One source (S1 and
   S2) says…"*: it leaked internal source ids, which the system instruction forbids.
   *Next:* a cross-encoder reranker over the top 10–20 hits, and a post-check that strips
   `S<n>` labels from answers.
3. **On-topic but unanswerable questions pass `MIN_SCORE`** (section 5). Correct today only
   because the LLM refuses. *Next:* tune the threshold on a larger labelled set, or use a
   relative rule (for example, refuse when the top score is close to the median score of
   unrelated questions), and keep the prompt rule as the final check.
4. **Duplicate citations.** Citations are per chunk, so one page can appear twice (e.g.
   eatwell-02 lists page 8 twice). *Next:* merge citations by file and page in the response.
5. **No keyword matching.** Pure vector search can rank exact terms (units, numbers, names)
   below paraphrases. *Next:* hybrid search (BM25 + vectors, which Qdrant supports with
   sparse vectors).

## 7. Latency and cost

**Latency (measured, single run, free tiers; rate-limit waiting excluded):**

| Step | Time |
|---|---|
| Index all 3 documents at 800 / 120 (66 chunks: extract, chunk, embed, store) | 5.9 s |
| Retrieve per question (embed the question + Qdrant search) | 0.73 s mean |
| Generate per answer (Groq `gpt-oss-120b`) | 5.3 s mean |

The query embedding call takes most of the retrieval time; Qdrant itself is fast at this
size. The free Gemini tier allows **100 embedded texts per minute**, so on the free tier a
document with more than 100 chunks takes at least a minute to index. The evaluation script
waits for the limit, and the app retries a 429 a few times and then fails the upload
clearly.

**Cost (ESTIMATE).** Tokens are estimated as characters ÷ 4, not taken from billing.
Prices are list prices found on 30 October 2026, from Google's
[Gemini Embedding announcement](https://developers.googleblog.com/en/gemini-embedding-available-gemini-api/)
and third-party Groq trackers such as
[morphllm.com/groq-pricing](https://www.morphllm.com/groq-pricing). Check
[groq.com/pricing](https://www.groq.com/pricing) and the Gemini pricing page before relying
on them.

| Item | Estimated tokens | Price (USD per 1M tokens) | Estimated cost |
|---|---|---|---|
| Index the 3 documents once (embeddings) | ~10,800 | $0.15 input | ~$0.0016 |
| One question: query embedding | ~15 | $0.15 input | negligible |
| One question: answer prompt (instructions + 5 chunks + question) | ~1,175 | $0.15 input | ~$0.00018 |
| One question: answer output (visible text) | ~23 | $0.60 output | ~$0.00001 |

That is roughly **$0.20 per 1,000 questions**, before hidden reasoning tokens.
`gpt-oss-120b` is a reasoning model, and its reasoning tokens are billed as output but are
not in these estimates. If each answer used about 500 reasoning tokens, the cost would rise
to roughly $0.50 per 1,000 questions. Follow-up questions add one small rewrite call
(`gpt-oss-20b`). Ingestion cost is tiny compared with question answering. This whole
evaluation ran on free tiers at $0.

## 8. Limitations and trade-offs

- **Small set:** 21 questions over 3 documents, written by me. One question is about 6
  percentage points of recall, so differences of one or two questions are not significant.
- **One run.** Retrieval is deterministic. Answers use temperature 0, but they can still
  vary slightly between runs.
- **Strict hit rule:** the whole evidence phrase must be in one chunk. This penalises
  small chunks that hold half of a fact (section 3). That is fair for answering, but it
  understates how close their retrieval was.
- **Substring fact check:** fine for numbers and short facts. It cannot judge longer
  explanations or catch a correct number in a wrong sentence; that would need a human or
  an LLM judge.
- **Not measured here:** follow-up questions and history (covered by the automated tests in
  `tests/test_history.py` and the live tests), document updates, and larger collections,
  where `TOP_K = 5` and the score threshold may behave differently.
- **Free-tier models** were used for both providers; latency on paid tiers will differ.
