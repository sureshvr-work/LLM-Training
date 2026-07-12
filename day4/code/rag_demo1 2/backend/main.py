"""
main.py  —  the RAG lab backend, one endpoint per stage of the pipeline.

The whole demo works on ONE document at a time, so we keep its state in a
single in-memory dict (STATE). This is a teaching demo for one user, so a
plain global is the simplest honest choice — no database, no sessions.

Stages (each is an endpoint the UI calls in order):
    /api/upload  -> parse the PDF into per-page text
    /api/chunk   -> fixed-size chunking (with overlap), tracking page numbers
    /api/embed   -> turn chunks into vectors + return a heatmap preview
    /api/search  -> embed the query, cosine vs every chunk, return top-K
    /api/answer  -> hand the retrieved chunks to the LLM, return the answer
"""
import bisect
import io
import re

import numpy as np
import pdfplumber
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from llm import embed_texts, embed_text, answer

app = FastAPI(title="RAG lab")

# ---- single-document state, in memory --------------------------------------
STATE = {
    "pages": [],        # list of per-page text strings
    "full": "",         # all pages joined into one string
    "starts": [],       # char offset where each page begins in `full`
    "chunks": [],       # list of {i, text, start, end, page_start, page_end}
    "vectors": None,    # numpy matrix (n_chunks x dim)
    "embed_model": None,
    "source": None,     # the uploaded filename — travels with every chunk as metadata
    "pca": None,        # {mean, comps} basis so the query can be projected into the same 2-D map
}

PREVIEW_DIMS = 48  # how many of the (e.g.) 1536 dimensions we show in the heatmap


def _page_of(char_index: int) -> int:
    """Which page (1-indexed) a character offset falls on."""
    return bisect.bisect_right(STATE["starts"], char_index)


# ---- STAGE 1: upload + parse -----------------------------------------------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    pages = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            # Drop the repeated running footer (e.g. "... Page 7 of 10"). pdfplumber
            # pulls it from every page, and otherwise it lands inside chunks as noise
            # and gets embedded as if it were content.
            lines = [ln for ln in text.split("\n")
                     if not re.search(r"Page\s+\d+\s+of\s+\d+\s*$", ln.strip())]
            pages.append("\n".join(lines))

    # Join pages into one string, remembering where each page starts.
    full = ""
    starts = []
    for i, text in enumerate(pages):
        starts.append(len(full))
        full += text
        if i < len(pages) - 1:
            full += "\n"

    STATE.update(pages=pages, full=full, starts=starts,
                 chunks=[], vectors=None, embed_model=None, source=file.filename)

    return {
        "filename": file.filename,
        "pages": len(pages),
        "total_chars": len(full),
        "per_page": [{"page": i + 1, "chars": len(t)} for i, t in enumerate(pages)],
    }


# ---- STAGE 2: fixed-size chunking ------------------------------------------
class ChunkRequest(BaseModel):
    size: int = 800
    overlap: int = 150


@app.post("/api/chunk")
def chunk(req: ChunkRequest):
    full = STATE["full"]
    size = max(50, req.size)
    overlap = min(max(0, req.overlap), size - 1)  # overlap must be < size
    step = size - overlap

    chunks = []
    start = 0
    i = 0
    while start < len(full):
        end = min(start + size, len(full))
        text = full[start:end]
        chunks.append({
            "i": i,
            "id": f"c{i + 1}",
            "text": text,
            "start": start,
            "end": end,
            "page_start": _page_of(start),
            "page_end": _page_of(max(start, end - 1)),
        })
        i += 1
        if end >= len(full):
            break
        start += step

    STATE.update(chunks=chunks, vectors=None, embed_model=None)

    # For each chunk, how many characters it SHARES with its neighbours.
    # head = chars repeated from the previous chunk (the leading overlap)
    # tail = chars that will repeat into the next chunk (the trailing overlap)
    # Computed from real offsets so they line up with the raw text exactly.
    n = len(chunks)
    out = []
    for idx, c in enumerate(chunks):
        head = (chunks[idx - 1]["end"] - c["start"]) if idx > 0 else 0
        tail = (c["end"] - chunks[idx + 1]["start"]) if idx < n - 1 else 0
        out.append({
            "id": c["id"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "source": STATE["source"],
            "chars": len(c["text"]),
            "overlap_head": max(0, head),
            "overlap_tail": max(0, tail),
            "text": c["text"],   # RAW (not stripped) so highlight offsets line up
        })

    return {
        "count": n,
        "size": size,
        "overlap": overlap,
        "step": step,
        "chunks": out,
    }


# ---- STAGE 3: embed + heatmap preview --------------------------------------
class EmbedRequest(BaseModel):
    model: str = "text-embedding-3-small"
    sdk: str = "openai"


@app.post("/api/embed")
def embed(req: EmbedRequest):
    texts = [c["text"] for c in STATE["chunks"]]
    vectors = embed_texts(texts, model=req.model, sdk=req.sdk)
    matrix = np.array(vectors, dtype=float)

    STATE.update(vectors=matrix, embed_model=req.model)

    dim = matrix.shape[1]
    preview = matrix[:, :PREVIEW_DIMS]
    scale = float(np.max(np.abs(preview))) or 1.0  # for color scaling on the UI

    rows = []
    for c, row in zip(STATE["chunks"], preview):
        rows.append({
            "id": c["id"],
            "page": c["page_start"],
            "cells": [round(float(v), 4) for v in row],  # raw values; UI colors them
        })

    # chunk x chunk cosine similarity — shows that semantic closeness lives in
    # vector DIRECTION, not in array position. Computed only for small N.
    sim = None
    top_pair = None
    n = matrix.shape[0]
    if 2 <= n <= 60:
        Mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
        Smat = Mn @ Mn.T
        sim = [[round(float(x), 3) for x in r] for r in Smat]
        bi, bj, bs = 0, 1, -2.0
        for i in range(n):
            for j in range(i + 1, n):
                if Smat[i, j] > bs:
                    bi, bj, bs = i, j, float(Smat[i, j])
        top_pair = {"i": bi, "j": bj, "score": round(bs, 3)}

    # 2-D semantic map: project the high-dim vectors down to (x, y) so each chunk
    # becomes a POINT. Near points = similar meaning. We keep the basis (mean +
    # the two top directions) so the query can be dropped into the SAME map later.
    pca_map = None
    STATE["pca"] = None
    if n >= 2:
        mean = matrix.mean(axis=0)
        Xc = matrix - mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:2] if Vt.shape[0] >= 2 else np.vstack([Vt, np.zeros((1, Vt.shape[1]))])
        coords = Xc @ comps.T
        STATE["pca"] = {"mean": mean, "comps": comps}
        pca_map = [
            {"id": STATE["chunks"][i]["id"], "page": STATE["chunks"][i]["page_start"],
             "x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4)}
            for i in range(n)
        ]

    return {
        "model": req.model,
        "sdk": req.sdk,
        "n_chunks": len(STATE["chunks"]),
        "dim": dim,                 # e.g. 1536
        "preview_dims": preview.shape[1],
        "scale": scale,
        "rows": rows,
        "labels": [{"id": c["id"], "page": c["page_start"]} for c in STATE["chunks"]],
        "sim": sim,
        "top_pair": top_pair,
        "map": pca_map,
    }


# ---- STAGE 4: search (cosine) ----------------------------------------------
def _cosine_all(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    qn = q / (np.linalg.norm(q) or 1.0)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return Mn @ qn


class SearchRequest(BaseModel):
    query: str
    top_k: int = 4
    sdk: str = "openai"


@app.post("/api/search")
def search(req: SearchRequest):
    M = STATE["vectors"]
    # Embed the query with the SAME model used for the chunks.
    q = np.array(embed_text(req.query, model=STATE["embed_model"], sdk=req.sdk), dtype=float)

    scores = _cosine_all(q, M)
    order = np.argsort(scores)[::-1][: req.top_k]  # best-first, keep top-K

    results = []
    for rank, idx in enumerate(order, 1):
        c = STATE["chunks"][int(idx)]
        results.append({
            "rank": rank,
            "id": c["id"],
            "page": c["page_start"],
            "source": STATE["source"],
            "score": round(float(scores[idx]), 4),
            "text": c["text"].strip(),
        })

    # Project the query into the SAME 2-D map (for the semantic-map overlay).
    query_xy = None
    if STATE.get("pca"):
        p = STATE["pca"]
        q2 = (q - p["mean"]) @ p["comps"].T
        query_xy = {"x": round(float(q2[0]), 4), "y": round(float(q2[1]), 4)}

    return {"query": req.query, "top_k": req.top_k, "results": results, "query_xy": query_xy}


# ---- STAGE 5: answer (LLM) -------------------------------------------------
class AnswerRequest(BaseModel):
    question: str
    top_k: int = 4
    sdk: str = "openai"
    model: str = "gpt-4o-mini"


@app.post("/api/answer")
def answer_endpoint(req: AnswerRequest):
    M = STATE["vectors"]
    q = np.array(embed_text(req.question, model=STATE["embed_model"], sdk=req.sdk), dtype=float)
    scores = _cosine_all(q, M)
    order = np.argsort(scores)[::-1][: req.top_k]

    matches = []
    context_items = []
    for rank, idx in enumerate(order, 1):
        c = STATE["chunks"][int(idx)]
        # Metadata travels INTO the prompt so the model can cite it.
        context_items.append({
            "id": c["id"], "page": c["page_start"], "source": STATE["source"], "text": c["text"],
        })
        matches.append({
            "rank": rank, "id": c["id"], "page": c["page_start"], "source": STATE["source"],
            "score": round(float(scores[idx]), 4), "text": c["text"].strip(),
        })

    llm_answer = answer(req.question, context_items, model=req.model, sdk=req.sdk)

    return {
        "answer": llm_answer,
        "sdk_used": req.sdk,
        "model": req.model,
        "matches": matches,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "pages": len(STATE["pages"]),
        "chunks": len(STATE["chunks"]),
        "embedded": STATE["vectors"] is not None,
    }
