# RAG Lab — watch every stage

A teaching app that exposes the whole RAG pipeline one stage at a time. Each step
renders what it produced, so the room sees a PDF become chunks become vectors
become a ranked search become an answer.

## Run
```bash
cp .env.example .env        # paste your OPENAI_API_KEY
docker compose up --build
```
Open http://localhost:8080 and click **Load the sample drug reference** (bundled),
or upload your own PDF.

## The 11 steps (mapped to the UI)
1. **Model & SDK** — choose the chat model and the SDK path: **OpenAI SDK (direct)** or **LangChain (OpenAI)**. The choice applies to both embedding and the final answer, so the contrast is real.
2. **Upload PDF** — parsed in the backend (pdfplumber); shows page count + chars-per-page.
3. **Chunk (fixed-size)** — size + overlap sliders.
4. **Chunk preview** — chunk count and every chunk, each with a page badge (chunks that span two pages show why overlap matters).
5. **Embedding model** — API-based; OpenAI `text-embedding-3-small` (1536-d) or `-large` (3072-d).
6. **Embed** — one click; runs via the chosen SDK.
7. **Vector heatmap** — each chunk is a row; the first 48 of the N dimensions as blue (neg) / red (pos) cells. Honest preview: you can't see all 1536, this is the shape.
8. **Query** — live echo of the search string as you type.
9. **top_k** — slider with a plain-language description.
10. **Retrieve (cosine)** — ranked chunks with cosine score bars + page badges.
11. **Send to LLM** — the retrieved chunks become the context; answered via your chosen SDK path.

## Structure
```
rag-lab/
├── docker-compose.yml
├── .env.example
├── frontend/   nginx: wizard UI + bundled sample PDF, proxies /api/* → backend:8000
│   ├── Dockerfile · nginx.conf · index.html · sample_drug_reference.pdf
├── backend/    FastAPI — one endpoint per stage, document state held in memory
│   ├── Dockerfile · requirements.txt · main.py
└── llm/        the only place that talks to a provider; OpenAI-direct OR LangChain
    ├── __init__.py · embed.py · chat.py
```

## Endpoints
`/api/upload` (PDF) · `/api/chunk` · `/api/embed` · `/api/search` · `/api/answer` · `/api/health`

The document state lives in one in-memory dict in the backend — a single-user
teaching demo, so no database or sessions. Retrieval is **dense/cosine only**;
keyword (BM25) and hybrid are a later session.
