# Hybrid Lab — dense + sparse in one page

A single staged lab where **one document** is retrieved **two ways** and fused:

1. **Model & SDK** — OpenAI for embeddings + answer (OpenAI SDK or LangChain path).
2. **Upload** — PDF or text, split into pages.
3. **Chunk once** — fixed-size windows with overlap. The *same* chunks feed both lanes.
4. **Build both** — side by side:
   - **Dense:** OpenAI embeddings → a 2-D PCA map + cosine.
   - **Sparse:** an inverted index (analyzer pipeline + postings).
5. **Search** — one query, three columns: **cosine**, **BM25** (with per-term math),
   and **RRF fusion** (`1/(k+dense_rank) + 1/(k+sparse_rank)`, k=60) showing how each
   lane's rank combined.
6. **Answer** — the fused (hybrid) top-k becomes the LLM context, with `[c3, p2]` citations.

One backend, one frontend, one compose.

```
hybrid-lab/
├─ docker-compose.yml        # 2 services: backend (8000) + frontend (8080)
├─ .env.example
├─ backend/
│  ├─ Dockerfile · requirements.txt · llm.py
│  └─ app/main.py            # upload · chunk · embed · index · search(3-lane) · answer
└─ frontend/
   ├─ Dockerfile · nginx.conf
   ├─ index.html             # the single combined wizard
   └─ sample_drug_reference.pdf
```

## Run
```bash
cp .env .env         # paste your OPENAI_API_KEY
docker compose up --build
```
Open **http://localhost:8080**. API docs at http://localhost:8000/docs.

## Try it
Load the sample drug reference → Chunk → **Embed** and **Build index** → ask
*"When do you reduce the apixaban dose?"* → Retrieve (watch the three lanes) → Send to LLM.
Notice the dense lane matches on meaning, the sparse lane on exact terms, and RRF lifts
whatever either lane ranks highly.

## Notes
- In-memory, single document (a teaching lab); restart clears state.
- BM25 k1=1.5, b=0.75; analyzer stems word *forms* only (reports→report), never meanings.
- The 2-D map is a numpy PCA (SVD) of the embeddings; the query is projected with the same components.
