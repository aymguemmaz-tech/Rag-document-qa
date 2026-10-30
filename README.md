# RAG Document Q&A — GPT-based retrieval with a measurable evaluation layer

[![CI](https://github.com/aymguemmaz-tech/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/aymguemmaz-tech/rag-document-qa/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.13-blue)
![pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791)
![License](https://img.shields.io/badge/license-MIT-green)

A production-style retrieval-augmented generation (RAG) system that answers questions about your
documents **with citations**, refuses to answer when the documents do not contain the answer, and
comes with an **evaluation layer that measures every design choice** instead of judging answers by eye.

- **Hybrid retrieval on PostgreSQL**: pgvector HNSW for meaning, Okapi **BM25 implemented in SQL** for
  exact terms, fused with Reciprocal Rank Fusion.
- **Grounded generation**: GPT (any OpenAI-compatible model) answers only from labelled sources, cites
  them as `[S1]`, abstains when unsure, ignores instructions planted in documents.
- **Evaluation as engineering**: a 150-question gold set with evidence spans, 25-experiment ablation
  grid, bootstrap confidence intervals, paired permutation tests with Holm correction, an output
  variability study, and a CI quality gate that fails the build when quality regresses.

![Web UI answering with citations](docs/images/web-ui.png)

---

## Results (reference run, test split)

The committed reference run ([results/offline/report.md](results/offline/report.md)) uses the fully
offline stack (WordLlama embeddings + extractive generator) so anyone can reproduce it in about 25 seconds
without an API key: 100 test questions over 21 documents, 95% bootstrap confidence intervals.

| Metric | Baseline (hybrid retrieval, structure chunking) |
| --- | --- |
| Recall@5 (evidence found in top 5) | **98.3%** [95.5, 100.0] |
| MRR | **91.0%** [86.1, 95.3] |
| Answer accuracy (answerable questions) | 80.7% [71.6, 88.6] |
| Correct abstention (unanswerable questions) | 41.7% [16.7, 66.7] |
| Citation precision | 93.9% [88.4, 98.2] |
| Prompt-injection attacks followed | 0 of 3 |
| Latency p50 (retrieval + generation, CPU) | 3.5 ms |

What the ablation showed (paired differences vs the baseline, same questions):

| Change | Effect | 95% CI | Holm-adjusted p |
| --- | --- | --- | --- |
| + MaxSim late-interaction reranker | MRR **+5.1 pp** (Recall@1 83.0 → 90.3%) | [+2.0, +8.7] | **0.043** |
| Dense only (no BM25) | MRR −3.5 pp, Recall@5 98.3 → 94.3% | [−6.7, −0.6] | 0.69 |
| No contextual chunk headers | MRR −3.5 pp | [−7.2, −0.2] | 0.82 |
| No superseded-document filter | MRR −1.9 pp, citation precision 93.9 → 90.2% | [−4.0, −0.2] | 1.00 |
| No injection guard | attacks followed 0 → 1 of 3 | – | – |
| Only 1 chunk of context (k = 1) | overall correct −7.0 pp | [−12, −2] | 0.42 |

![Effect of each change on MRR](results/offline/charts/effects_mrr.png)

**Reading the numbers honestly.** Only the MaxSim reranker survives correction for 18 comparisons; the
other effects are consistent in direction but not proven on 100 questions. Retrieval is no longer the
bottleneck (98% of evidence is retrieved) — answering is: the extractive baseline abstains correctly on
only 5 of 12 unanswerable questions and struggles with multi-hop and paraphrased questions. That is
exactly what the GPT generator with the `strict` prompt is for; `make eval-gpt` runs the same grid with
GPT, prompt ablations and an LLM judge (roughly $1 of API usage with `gpt-6-luna`).

---

## Architecture

```mermaid
flowchart LR
  subgraph Ingestion
    A[Markdown / PDF / HTML / TXT] --> B[Loaders<br/>sections + char offsets]
    B --> C[Chunkers<br/>fixed · recursive · semantic · structure]
    C --> D[Embedders<br/>OpenAI · WordLlama · hashing<br/>+ SQLite cache]
    D --> E[(PostgreSQL + pgvector<br/>HNSW · BM25 in SQL)]
  end
  subgraph Query
    Q[Question] --> X[Query expansion<br/>multi-query · HyDE]
    X --> R1[Dense search]
    X --> R2[BM25 search]
    R1 --> F[Reciprocal Rank Fusion]
    R2 --> F
    F --> G[Superseded filter<br/>+ injection guard]
    G --> H[Rerank<br/>MaxSim · MMR · LLM · cross-encoder]
    H --> I[Context builder<br/>token budget · S# labels]
    I --> J[Generator<br/>GPT · extractive]
    J --> K[Citation check +<br/>grounding verifier]
    K --> L[Answer · sources · trace]
  end
  E --- R1
  E --- R2
  subgraph Evaluation
    M[Gold set<br/>150 Q + evidence spans] --> N[Ablation runner]
    N --> O[Metrics · bootstrap CIs<br/>permutation tests · Holm]
    O --> P[Report · charts · CI gate]
  end
  N -. drives .-> Q
```

| Layer | Highlights |
| --- | --- |
| Ingestion | Markdown heading paths, PDF page numbers, front-matter lifecycle metadata (`status: superseded`), content-hash incremental re-indexing, persistent embedding cache |
| Chunking | 4 strategies on exact character spans; contextual headers ("Doc › Section") prepended to the embedded text |
| Storage | PostgreSQL/pgvector (HNSW, cosine) with one table per collection; in-memory store with BM25 for lite mode and tests |
| Retrieval | dense, BM25 or hybrid (RRF); query expansion; document-lifecycle filter; prompt-injection guard; 4 rerankers |
| Generation | versioned prompt templates, `[S#]` citations, abstention, streaming (SSE), model-parameter negotiation, token + cost accounting |
| Safety | injection detector (drop/flag), numeric-consistency grounding check, citation validation, API key auth |
| Serving | FastAPI (OpenAPI docs, SSE, uploads, stats), web UI, Typer CLI, Docker + compose, structured logs with request ids |
| Quality | 73 tests (unit + pgvector integration, 92% coverage), ruff, mypy, GitHub Actions with an evaluation quality gate |

---

## Quickstart

```bash
git clone https://github.com/aymguemmaz-tech/rag-document-qa.git
cd rag-document-qa
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,eval]"
```

**1. Lite mode — no API key, no database** (WordLlama embeddings + extractive answers):

```bash
ragqa ingest data/corpus
ragqa ask "How long is identifiable GPS trip data kept?"
ragqa serve                       # web UI + API on http://127.0.0.1:8000 (docs at /docs)
```

**2. GPT mode** (OpenAI embeddings + `gpt-6-luna`; any OpenAI-compatible server works via `RAGQA_LLM_BASE_URL`):

```bash
export OPENAI_API_KEY=sk-...
ragqa ingest data/corpus --collection gpt      # a new collection: different embedder, different vectors
ragqa ask "What caused the April payment outage?" --collection gpt
```

**3. Production setup with PostgreSQL + pgvector:**

```bash
docker compose up --build -d
docker compose run --rm api ragqa ingest data/corpus
open http://localhost:8000
```

Example (lite mode; `--json` returns the full answer object):

```text
$ ragqa ask "What caused the April payment outage?"
The automatic certificate renewal job had been disabled during the February infrastructure migration
and was never re-enabled [S2]. An intermediate TLS certificate on the connection between the ride
service and our payment gateway provider, PayLane, had expired [S2].

    [S1] Postmortem: Payment Authorisation Outage (14 April 2026) › Summary  (score 0.032)
  * [S2] Postmortem: Payment Authorisation Outage (14 April 2026) › Root cause  (score 0.032)
    ...
grounded=1.00  flags=['injection_flagged:1']  generator=extractive  total=13 ms  cost=$0.00000
```

`injection_flagged:1` means the guard removed a retrieved chunk that contained a planted instruction.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/query` | Answer with citations, sources, grounding score, timings, usage and cost |
| `POST` | `/v1/query/stream` | Same, as Server-Sent Events: `sources` → `token`… → `done` |
| `POST` | `/v1/documents` | Upload and index files (multipart; incremental) |
| `GET` / `DELETE` | `/v1/documents[/{id}]` | List or delete indexed documents |
| `GET` | `/v1/stats` | Requests, errors, abstention rate, latency p50/p95/p99, tokens, cost |
| `GET` | `/v1/config` | Effective pipeline configuration and its `config_id` |
| `GET` | `/health` | Liveness + index size |

```bash
curl -s localhost:8000/v1/query -H 'content-type: application/json' \
     -d '{"question": "Which extinguisher is used on a burning battery?"}' | jq .answer
```

Set `RAGQA_API_KEY` to require an `X-API-Key` header on all `/v1` endpoints.

---

## Evaluation layer

The sample corpus is a **fictional** company (Veloria Mobility, an e-scooter operator): 21 documents
including policies, a postmortem, quarterly reports, a PDF, two **superseded** document versions with
conflicting numbers and a forum page containing a **prompt-injection attack**. Because the facts are
invented, a model cannot answer from memory — the benchmark measures grounding, not recall of the internet.

- **150 questions** (factoid, numeric, paraphrase, multi-hop, unanswerable, injection), split 50 dev / 100 test.
- **Evidence spans, not chunk ids**: each answer cites the exact sentence it comes from; a retrieved chunk
  counts as a hit when it overlaps that span. This makes completely different chunking strategies comparable.
- **Metrics**: Hit/Recall/nDCG@k and MRR; accuracy, token F1, abstention; citation precision/recall;
  groundedness; injection followed; latency and cost.
- **Statistics**: 95% bootstrap intervals, paired sign-flip permutation tests, Holm–Bonferroni correction.
- **Variability study**: each question answered 5× per temperature — at T ≥ 0.25, 9–11% of questions
  flip between right and wrong while mean accuracy moves by less than 1 point.
- **Hygiene**: the abstention threshold is calibrated on the dev split only; results are reported on test.

```bash
ragqa eval validate                                   # every evidence span exists verbatim in the corpus
ragqa eval calibrate --split dev                      # tune the abstention threshold on dev only
ragqa eval run --spec configs/ablation.yaml           # offline grid -> results/offline/report.md + charts
ragqa eval run --spec configs/ablation_gpt.yaml       # GPT grid: prompts, query expansion, LLM rerank, judge
ragqa eval gate --config configs/quality_gate.yaml    # what CI runs on every push
```

---

## Configuration

Infrastructure comes from environment variables (see [.env.example](.env.example)); pipeline behaviour
from YAML ([configs/default.yaml](configs/default.yaml)), which is what the ablations vary.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | – | Enables OpenAI embeddings and the GPT generator (`auto` mode) |
| `RAGQA_LLM_MODEL` | `gpt-6-luna` | Chat model; `RAGQA_LLM_BASE_URL` for Ollama, vLLM or other compatible servers |
| `RAGQA_EMBEDDER` / `RAGQA_GENERATOR` | `auto` | Force `openai`/`wordllama`/`hashing` and `llm`/`extractive` |
| `RAGQA_STORE` | `memory` | `memory` (persisted in `RAGQA_DATA_DIR`) or `pgvector` |
| `RAGQA_DATABASE_URL` | `postgresql://ragqa:ragqa@localhost:5432/ragqa` | PostgreSQL with the `vector` extension |
| `RAGQA_PIPELINE_CONFIG` | – | Pipeline YAML (chunking, retrieval, generation, guards) |

---

## Project structure

```text
src/ragqa/
├── ingest/        loaders (md, txt, html, pdf) and chunking strategies
├── embeddings.py  OpenAI / WordLlama / hashing embedders + SQLite cache
├── store/         VectorStore interface, in-memory store, PostgreSQL/pgvector store
├── retrieval/     BM25, RRF fusion, rerankers, query expansion, retriever
├── generation/    prompts, LLM client, generators, citations, guards
├── pipeline.py    ingestion + question answering, assembled from config
├── api/           FastAPI app, schemas, web UI
├── eval/          gold set, metrics, statistics, ablation runner, variability, report, LLM judge
└── cli.py         `ragqa` command
configs/           pipeline + ablation + quality-gate YAML
data/              fictional corpus and gold evaluation set
results/offline/   reference evaluation run (report, CSV/JSON, charts)
tests/             unit and pgvector integration tests
```

## Design decisions

- **BM25 in SQL** — Postgres `ts_rank` has no inverse document frequency, so rare, decisive terms (codes,
  names) are under-weighted; BM25 is computed from `tsvector` statistics with GIN-indexed candidates.
- **One table per collection** — every collection gets its own fixed-dimension HNSW index, so filtering by
  collection can never truncate approximate-nearest-neighbour results.
- **Reciprocal Rank Fusion** — combines cosine and BM25 rankings without calibrating their scales.
- **Deterministic chunk sizing** — chunks are measured in regex tokens, so results reproduce on any machine.
- **Provider-agnostic LLM client** — Chat Completions plus automatic negotiation of unsupported parameters
  (e.g. `temperature` on reasoning models) lets one config run on OpenAI, Ollama or vLLM.
- **An offline floor** — the extractive generator makes the full evaluation runnable in CI for free and
  gives GPT a baseline it has to beat to justify its cost.

## Limitations and next steps

- The extractive baseline is weak on abstention, paraphrase and multi-hop questions — run the GPT grid to
  quantify how much the LLM recovers.
- The corpus is small and keyword-heavy, which favours BM25; a larger, noisier corpus would stress dense retrieval.
- The grounding verifier is a fast heuristic; the LLM judge (`judge: true`) should be spot-checked against it.
- Next: cross-encoder reranking on GPU, semantic caching of answers, per-user document permissions.

## Author

Aymen Charef Eddine Guemmaz — [LinkedIn](https://www.linkedin.com/in/aymen-charef-eddine-guemmaz-17a118329) ·
[GitHub](https://github.com/aymguemmaz-tech). Released under the [MIT License](LICENSE).
