"""
main.py — the API for the pgvector teaching demo.

Endpoints (all under /api):
  GET  /api/health                 -> {ok}
  GET  /api/status                 -> lifecycle: extension? table? rows? index?
  POST /api/setup                  -> CREATE EXTENSION -> CREATE TABLE -> embed+insert -> CREATE INDEX (returns each SQL + result)
  POST /api/reset                  -> DROP TABLE docs (keeps the extension)
  GET  /api/docs                   -> [{id,title}] for the picker
  GET  /api/doc/{id}               -> one row + what its vector looks like
  POST /api/search {q, metric}     -> BOTH lanes: keyword (exact) and vector (meaning)
  POST /api/insert {title, body}   -> embed + insert one new doc
  GET  /api/selftest               -> cosine similarity of a few sentence pairs (proves embeddings work)

The frontend (single page) is served from / .
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

from psycopg.rows import dict_row

import embed
import db
from seed import DOCS, EXAMPLE_QUERIES

app = FastAPI(title="pgvector demo")

DIM = embed.DIM

# operator per metric, and how to turn the operator's value into a friendly score
METRICS = {
    "cosine": {"op": "<=>", "label": "cosine distance"},
    "l2":     {"op": "<->", "label": "L2 distance"},
    "ip":     {"op": "<#>", "label": "inner product"},
}

CREATE_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS docs (\n"
    "  id        serial PRIMARY KEY,\n"
    "  title     text,\n"
    "  body      text,\n"
    f"  embedding vector({DIM})        -- the vector column\n"
    ")"
)
CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS docs_embedding_hnsw\n"
    "  ON docs USING hnsw (embedding vector_cosine_ops)"
)


# ----------------------------------------------------------------------------- helpers
def _status(conn):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_extension WHERE extname='vector'")
    ext = cur.fetchone() is not None
    cur.execute("SELECT to_regclass('public.docs') IS NOT NULL")
    table = cur.fetchone()[0]
    rows = 0
    index = False
    if table:
        cur.execute("SELECT count(*) FROM docs")
        rows = cur.fetchone()[0]
        cur.execute("SELECT 1 FROM pg_indexes WHERE tablename='docs' AND indexdef ILIKE '%hnsw%'")
        index = cur.fetchone() is not None
    return {"extension": ext, "table": bool(table), "rows": rows, "index": index}


# ----------------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/status")
def status():
    with db.pool().connection() as conn:
        return _status(conn)


@app.post("/api/setup")
def setup():
    steps = []
    with db.pool().connection() as conn:
        cur = conn.cursor()

        # 1) extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        steps.append({"title": "1 · Install the extension",
                      "sql": "CREATE EXTENSION IF NOT EXISTS vector;",
                      "detail": "vector type + ANN index methods are now available in this database."})

        # 2) table
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        steps.append({"title": "2 · Create the table",
                      "sql": CREATE_TABLE_SQL + ";",
                      "detail": f"Normal columns (title, body) live beside a vector({DIM}) column."})

        # 3) embed + insert seed (only if empty)
        cur.execute("SELECT count(*) FROM docs")
        existing = cur.fetchone()[0]
        if existing == 0:
            vecs = embed.embed_texts([b for _, b in DOCS])
            for (title, body), v in zip(DOCS, vecs):
                cur.execute(
                    "INSERT INTO docs (title, body, embedding) VALUES (%s, %s, %s)",
                    (title, body, np.asarray(v, dtype=np.float32)),
                )
            conn.commit()
            steps.append({"title": "3 · Embed & store the rows",
                          "sql": "INSERT INTO docs (title, body, embedding)\nVALUES (%s, %s, %s);   -- embedding = embed(body)",
                          "detail": f"Each body was run through the model → a {DIM}-dim vector, then stored. ({len(DOCS)} rows)"})
        else:
            steps.append({"title": "3 · Embed & store the rows",
                          "sql": "-- skipped: rows already present",
                          "detail": f"{existing} rows already stored."})

        # 4) ANN index
        cur.execute(CREATE_INDEX_SQL)
        conn.commit()
        steps.append({"title": "4 · Build the ANN index (HNSW)",
                      "sql": CREATE_INDEX_SQL + ";",
                      "detail": "HNSW makes nearest-neighbour search fast (approximate). cosine opclass = use <=>."})

        st = _status(conn)
    return {"steps": steps, "status": st}


@app.post("/api/reset")
def reset():
    with db.pool().connection() as conn:
        conn.cursor().execute("DROP TABLE IF EXISTS docs")
        conn.commit()
        return {"ok": True, "status": _status(conn)}


@app.get("/api/docs")
def list_docs():
    with db.pool().connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        try:
            cur.execute("SELECT id, title FROM docs ORDER BY id")
            return {"docs": cur.fetchall()}
        except Exception:
            return {"docs": []}


@app.get("/api/doc/{doc_id}")
def get_doc(doc_id: int):
    with db.pool().connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute("SELECT id, title, body, embedding FROM docs WHERE id=%s", (doc_id,))
        r = cur.fetchone()
    if not r:
        return JSONResponse({"error": "not found"}, status_code=404)
    v = np.asarray(r["embedding"], dtype=np.float32)
    return {
        "id": r["id"], "title": r["title"], "body": r["body"],
        "dim": int(v.shape[0]),
        "preview": [round(float(x), 4) for x in v[:8].tolist()],
        "norm": round(float(np.linalg.norm(v)), 4),
    }


class SearchIn(BaseModel):
    q: str
    metric: str = "cosine"
    k: int = 5


@app.post("/api/search")
def search(inp: SearchIn):
    q = (inp.q or "").strip()
    metric = inp.metric if inp.metric in METRICS else "cosine"
    op = METRICS[metric]["op"]
    k = max(1, min(10, inp.k))

    with db.pool().connection() as conn:
        st = _status(conn)
        if not st["table"]:
            return {"error": "no table yet — click Run setup first.", "status": st}

        cur = conn.cursor(row_factory=dict_row)

        # ---- lane A: NORMAL SQL, exact text match ----
        kw_sql_shown = ("SELECT id, title, body\n"
                        "FROM docs\n"
                        f"WHERE title ILIKE '%{q}%' OR body ILIKE '%{q}%';")
        like = f"%{q}%"
        cur.execute("SELECT id, title, body FROM docs WHERE title ILIKE %s OR body ILIKE %s ORDER BY id",
                    (like, like))
        keyword_rows = cur.fetchall()

        # ---- lane B: VECTOR search, by meaning ----
        qvec = np.asarray(embed.embed_one(q), dtype=np.float32) if q else None
        vec_rows = []
        vec_sql_shown = (
            "SELECT id, title, body,\n"
            f"       embedding {op} $q AS distance      -- $q = embed('{q}')\n"
            "FROM docs\n"
            f"ORDER BY embedding {op} $q\n"
            f"LIMIT {k};"
        )
        if q:
            cur.execute(
                f"SELECT id, title, body, embedding {op} %s AS distance "
                f"FROM docs ORDER BY embedding {op} %s LIMIT %s",
                (qvec, qvec, k),
            )
            for r in cur.fetchall():
                d = float(r["distance"])
                if metric == "cosine":
                    sim = round(1.0 - d, 3)            # cosine similarity
                elif metric == "ip":
                    sim = round(-d, 3)                  # <#> returns negative inner product
                else:
                    sim = None                         # L2: smaller distance = closer
                vec_rows.append({"id": r["id"], "title": r["title"], "body": r["body"],
                                 "distance": round(d, 4), "similarity": sim})

    return {
        "query": q,
        "metric": metric, "metric_label": METRICS[metric]["label"], "op": op,
        "keyword": {"sql": kw_sql_shown, "rows": keyword_rows},
        "vector": {"sql": vec_sql_shown, "rows": vec_rows,
                   "qpreview": [round(float(x), 3) for x in qvec[:6].tolist()] if q else []},
        "status": st,
    }


class InsertIn(BaseModel):
    title: str
    body: str


@app.post("/api/insert")
def insert(inp: InsertIn):
    title = (inp.title or "").strip() or "(untitled)"
    body = (inp.body or "").strip()
    if not body:
        return JSONResponse({"error": "body is required"}, status_code=400)
    v = np.asarray(embed.embed_one(body), dtype=np.float32)
    with db.pool().connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO docs (title, body, embedding) VALUES (%s,%s,%s) RETURNING id",
                        (title, body, v))
        except Exception as e:
            return JSONResponse({"error": f"insert failed — run setup first? ({e})"}, status_code=400)
        new_id = cur.fetchone()[0]
        conn.commit()
    return {"id": new_id, "dim": int(v.shape[0]),
            "preview": [round(float(x), 4) for x in v[:8].tolist()]}


@app.get("/api/selftest")
def selftest():
    """Prove embeddings capture meaning: similar sentences score high, unrelated low."""
    pairs = [
        ("heart attack", "myocardial infarction"),
        ("money laundering", "anti-money-laundering reporting"),
        ("heart attack", "a warm cup of coffee"),
    ]
    out = []
    for a, b in pairs:
        va, vb = embed.embed_texts([a, b])
        va = np.asarray(va, dtype=np.float32); vb = np.asarray(vb, dtype=np.float32)
        cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
        out.append({"a": a, "b": b, "cosine_similarity": round(cos, 3)})
    return {"model": embed.MODEL_NAME, "dim": DIM, "pairs": out,
            "note": "Related pairs score high; the unrelated pair scores low. That's why vector search finds by meaning."}


@app.get("/api/examples")
def examples():
    return {"queries": EXAMPLE_QUERIES}


# serve the single-page frontend (mounted last so /api/* wins)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
