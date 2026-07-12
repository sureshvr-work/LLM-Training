# Postgres as a Vector DB — `pgvector` lab

A tiny full-stack demo so students go from **installing** a vector DB to **operating** it, and feel the
difference between **exact SQL matching** and **vector search by meaning** — on the *same* query.

Real **Postgres + pgvector** (no simulation), a **FastAPI** backend, and a single-page UI.

---

## Run it (2 commands)

```bash
docker compose build      # one-time; downloads the embedding model INTO the image
docker compose up
```

Then open **http://localhost:8095**

> Build needs internet once (Python packages + the embedding model). After that the running
> containers need no internet. **Build it before class** so the room only runs `docker compose up`.

The database is also exposed so students can poke it directly:

```bash
psql postgresql://postgres:postgres@localhost:5433/postgres
```

---

## The 5-minute teaching flow

1. **Install** — click **Run setup**. Watch the four real SQL statements scroll by:
   `CREATE EXTENSION vector` → `CREATE TABLE … vector(384)` → embed & `INSERT` → `CREATE INDEX … USING hnsw`.
   The four pills at the top turn green. *"That's the entire installation — same Postgres, one extension."*

2. **Self-test embeddings** — click it. "heart attack" vs "myocardial infarction" scores high; vs "coffee"
   scores low. *"This is why vector search understands meaning."*

3. **The money shot** — type **`heart attack`** and Search:
   - **Normal SQL (left)** → `0 rows` — no document literally says "heart attack".
   - **Vector search (right)** → the *myocardial infarction* row comes back first, by meaning.
   Try the chips: `money laundering`, `can't log in`, `send a product back`. Keyword keeps missing; vectors keep finding.

4. **Store new data** — add your own title/body. It's embedded (`text → 384-dim vector`) and stored, then
   immediately searchable. Shows the extra step vector data always has.

5. **Peek at a vector** — pick a row to see what "vector data" actually is: a 384-dim float array, `‖v‖≈1`.

6. **Differences panel** — the normal-vs-vector cheat sheet to wrap up.

Switch the **operator** dropdown (`cosine <=>`, `L2 <->`, `inner product <#>`) to show there's a family of
distance measures — and that on normalized embeddings cosine and inner product rank the same.

---

## Try the SQL yourself (in `psql`)

```sql
\dx                                   -- see the vector extension installed
\d docs                               -- the table: note the  embedding vector(384)  column

-- normal: exact match (often misses)
SELECT title FROM docs WHERE body ILIKE '%heart attack%';

-- vector: nearest by meaning  (replace the array with a real embedding from the app)
SELECT title, embedding <=> '[…]' AS distance
FROM docs ORDER BY embedding <=> '[…]' LIMIT 5;

\di                                   -- the HNSW index
```

---

## What's inside

```
docker-compose.yml         db (pgvector/pgvector:pg16) + app (FastAPI)
backend/
  main.py                  API: setup, status, search (keyword + vector), insert, doc, selftest
  db.py                    connection pool + registers the pgvector adapter
  embed.py                 fastembed (BAAI/bge-small-en-v1.5, 384-dim) — the text→vector step
  seed.py                  the mixed finance/health/general corpus
  Dockerfile               bakes the model into the image
frontend/index.html        the single-page UI
```

## Knobs
- **Embedding model / dimension** — change `MODEL_NAME`/`DIM` in `embed.py` (and `vector(DIM)` follows).
- **Index** — swap HNSW for IVFFlat in `CREATE_INDEX_SQL` (`main.py`) to compare build time vs recall.
- **Ports** — app `8095`, db `5433` (edit `docker-compose.yml`).
