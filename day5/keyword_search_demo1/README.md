# Keyword Lab — watch every stage of the inverted index

The **sparse / keyword** counterpart to the dense RAG Lab. Same shape — one
document, walked one stage at a time — but the machine is the opposite one:

> **Dense lab:** text → *vector* → search by **cosine** (needs an embedding model).
> **Keyword lab:** text → *terms* → an **inverted index** → search by **BM25** (no model at all).

There is **no embedding model and no API call** in the retrieval path. The only
LLM call is the final answer step (and the optional LLM-driven chunker). That
absence *is* the lesson: keyword search is cheap and instant because it never
leaves the realm of exact strings — which is also why it goes blind to synonyms.

## Run

```bash
cp .env.example .env        # paste your OPENAI_API_KEY (only the answer step uses it)
docker compose up --build
```

Open **http://localhost:8090** and click **Load the banking compliance sample**,
upload your own PDF/text, or paste text. (Port 8090, so it can run alongside the
dense lab on 8080.)

## The six stages (mapped to the UI)

1. **Model & SDK** — for the *answer* only; OpenAI SDK direct or LangChain. Keyword retrieval needs neither.
2. **Load a document** — PDF (pdfplumber) or plain text, split into pages so chunks carry page numbers.
3. **Pick a chunking strategy** — the families that apply to an inverted index (below). Then inspect the chunks.
4. **Build the inverted index** — the analyzer (normalize → tokenize → stopwords → stem) turns each chunk into terms; every term points to the chunks that contain it (with term-frequency). Toggle stopwords/stemming and watch the index change.
5. **Ask & retrieve (BM25)** — the query runs through the **same analyzer** as the index, then BM25 scores every chunk. You see the per-term `idf × tf` breakdown and which query terms hit or missed the vocabulary.
6. **Send to the LLM** — the retrieved chunks become the context. (Hierarchical returns the *parent* chunk.)

## Chunking strategies — and the one that's missing

The **cut is engine-agnostic**: the same chunk you would embed, you also index.
Six of the seven taxonomy families apply here:

| Strategy | Family | What it does for the index |
|---|---|---|
| Fixed-size window | Fixed / rule-based | Char window + overlap. |
| Structure-aware | Structure-aware | Recursive split on blank lines → lines → sentences. |
| Meaning-based (lexical) | Meaning-based | Groups adjacent sentences by **shared vocabulary** (TextTiling-style) — a no-model proxy for semantic chunking. |
| Hierarchical | Hierarchical | Indexes small **child** chunks; returns the larger **parent** for context. |
| Enrichment (contextual) | Enrichment | Prepends each chunk with `[document — section]` **before indexing**, so a chunk that says only `$10,000` still answers “AML threshold.” Boosts BM25, not just embeddings. |
| LLM-driven | LLM-driven | Asks the model to split the document into titled sections (one API call). |

**Not here — Embedding-time** (late chunking, ColBERT): these live in vector
space and have **no inverted-index form**. They are the one family that can't
cross over, so the lab omits them by design.

## Structure

```
kw_demo1/
├── docker-compose.yml
├── .env.example
├── frontend/   nginx: the wizard + bundled sample, proxies /api/* → backend:8000
│   ├── Dockerfile · nginx.conf · index.html · sample_banking_reference.txt
├── backend/    FastAPI — one endpoint per stage
│   ├── Dockerfile · requirements.txt
│   ├── main.py    the HTTP wiring + in-memory document state
│   └── engine.py  the pure-Python keyword engine (analyzer, 6 chunkers, index, BM25) — no deps, unit-testable
└── llm/         the only place that calls a model: answer() + segment() (OpenAI direct OR LangChain)
```

## Endpoints

`/api/upload` (PDF) · `/api/upload_text` · `/api/strategies` · `/api/chunk` ·
`/api/index` · `/api/search` · `/api/answer` · `/api/health`

State lives in one in-memory dict in the backend — a single-user teaching demo,
so no database or sessions. Retrieval is **sparse/BM25 only**; the dense lab is
the cosine counterpart, and hybrid (fusing the two) is a later session.
