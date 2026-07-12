"""
Hybrid Lab backend — ONE document, TWO retrievers, ONE answer.

The same fixed-size chunks feed both lanes:
  dense  : OpenAI embeddings -> cosine similarity (+ a 2-D PCA map)
  sparse : an inverted index -> BM25
Search runs both and fuses them with Reciprocal Rank Fusion (RRF, k=60).
State is in memory, one document at a time (a teaching lab).
"""
import io, re, math
from typing import List, Dict, Optional
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import llm

PREVIEW_DIMS = 48
K1, B = 1.5, 0.75
RRF_K = 60

STOPWORDS = set("""a an the to of in on at for and or is are was were be been being this that these those
with without from by as it its your you we our their his her they them he she him i me my mine us do does
did done how what when where why which who whom will would can could should may might must not no nor so
than then there here over under up down out off into onto about above below within if else also any all
each per via using use used""".split())

app = FastAPI(title="Hybrid Lab backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE: Dict = {
    "filename": None, "pages": [], "full": "", "offsets": [], "chunks": [],
    # dense
    "vectors": None, "model": None, "pca_mean": None, "pca_comps": None,
    # sparse
    "index": {}, "df": {}, "avgdl": 0.0, "stop": True, "stem": True,
}


# ---------------- text / pages ----------------
def set_pages(pages: List[str], filename: str):
    pages = [p or "" for p in pages]
    full, offsets = "", []
    for i, pg in enumerate(pages):
        s = len(full); full += pg; e = len(full)
        offsets.append((s, e, i + 1)); full += "\n"
    STATE.update(filename=filename, pages=pages, full=full, offsets=offsets, chunks=[],
                 vectors=None, model=None, pca_mean=None, pca_comps=None,
                 index={}, df={}, avgdl=0.0)
    return {"filename": filename, "pages": len(pages),
            "total_chars": sum(len(p) for p in pages),
            "per_page": [{"page": i + 1, "chars": len(p)} for i, p in enumerate(pages)]}

def page_of(pos: int) -> int:
    for s, e, pg in STATE["offsets"]:
        if s <= pos < e: return pg
    return STATE["offsets"][-1][2] if STATE["offsets"] else 1

def paginate(text: str, approx: int = 1500) -> List[str]:
    paras = re.split(r"\n\s*\n", text.strip()); pages, cur = [], ""
    for p in paras:
        if len(cur) + len(p) > approx and cur:
            pages.append(cur.strip()); cur = ""
        cur += p + "\n\n"
    if cur.strip(): pages.append(cur.strip())
    return pages or [text.strip() or ""]


# A curated 4-chunk teaching corpus where BM25 and cosine each fail differently:
#   c1 topically near but vague (a cosine trap) · c2 keyword-stuffed wrong answer (a BM25 trap)
#   c3 the real answer (few query words, so BM25 buries it) · c4 an unrelated distractor
SCENARIO = {
    "filename": "teaching_scenario.txt",
    "query": "which blood thinners need a lower dose in frail elderly patients",
    "answer_id": "c3",
    "chunks": [
        "General background. Anticoagulants prevent the formation of blood clots and are widely used across cardiology and stroke prevention. Dosing varies by drug and by individual patient factors.",
        "Patient FAQs. Which blood thinners do frail elderly patients ask about, and which drugs need a lower dose when they feel unwell? Many patients want to know which medication to lower for an upset stomach.",
        "A factor Xa inhibitor should be given at the reduced regimen of 2.5 mg twice daily when at least two apply: age 80 years or older, body weight 60 kilograms or less, serum creatinine 1.5 mg/dL or higher.",
        "Warfarin requires routine INR testing with a target of 2 to 3, unlike the newer agents. Liver function and bleeding signs are checked at each visit.",
    ],
}

@app.post("/api/scenario")
def load_scenario():
    full = "\n\n".join(SCENARIO["chunks"])
    STATE.update(filename=SCENARIO["filename"], pages=[full], full=full, offsets=[(0, len(full) + 1, 1)],
                 vectors=None, model=None, pca_mean=None, pca_comps=None, index={}, df={}, avgdl=0.0)
    STATE["chunks"] = [{"id": f"c{i+1}", "page_start": 1, "page_end": 1, "chars": len(t),
                        "text": t, "overlap_head": 0, "overlap_tail": 0}
                       for i, t in enumerate(SCENARIO["chunks"])]
    return {"count": len(STATE["chunks"]), "size": 0, "overlap": 0, "step": 0,
            "chunks": STATE["chunks"], "scenario": True,
            "suggested_query": SCENARIO["query"], "answer_id": SCENARIO["answer_id"]}


# ---------------- analyzer (sparse) ----------------
def tokenize(t): return re.findall(r"[a-z0-9]+", re.sub(r"(?<=\d),(?=\d)", "", t.lower()))
def stem(w):
    for suf, rep in [("sses","ss"),("ies","y"),("ingly",""),("edly",""),("ing",""),
                     ("ed",""),("ment",""),("ness",""),("ly",""),("es",""),("s","")]:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)] + rep
    return w
def analyze(t, stop, do_stem):
    toks = tokenize(t)
    if stop: toks = [x for x in toks if x not in STOPWORDS]
    if do_stem: toks = [stem(x) for x in toks]
    return toks
def analyze_pipeline(t, stop, do_stem):
    lowered = tokenize(t)
    stopped = [x for x in lowered if stop and x in STOPWORDS]
    after = [x for x in lowered if not (stop and x in STOPWORDS)]
    final = [stem(x) if do_stem else x for x in after]
    return {"lowered": lowered, "stopped": stopped, "after_stop": after, "final": final}


# ---------------- request models ----------------
class TextUpload(BaseModel):
    text: str; filename: str = "pasted.txt"
class ChunkReq(BaseModel):
    size: int = 800; overlap: int = 150
class EmbedReq(BaseModel):
    model: str = "text-embedding-3-small"; sdk: str = "openai"
class IndexReq(BaseModel):
    stop: bool = True; stem: bool = True; sample_id: Optional[str] = None
class SearchReq(BaseModel):
    query: str
    retrieve_k: int = 10      # candidates each lane returns (and the pool RRF fuses)
    rrf_k: int = 60           # RRF smoothing constant (NOT a count)
    top_n: int = 5            # fused results to keep/show
    sdk: str = "openai"
class RerankReq(BaseModel):
    query: str
    candidates: List[Dict] = []   # [{id, dense_rank, sparse_rank, rrf_rank}] from /api/search
    top_n: int = 5
    model: str = "gpt-4o-mini"
    sdk: str = "openai"
class AnswerReq(BaseModel):
    question: str
    chunk_ids: Optional[List[str]] = None   # if given (e.g. reranked ids), answer from exactly these
    top_n: int = 5
    retrieve_k: int = 10
    rrf_k: int = 60
    sdk: str = "openai"
    model: str = "gpt-4o-mini"


# ---------------- upload ----------------
@app.get("/api/health")
def health(): return {"ok": True, "doc": STATE["filename"], "chunks": len(STATE["chunks"]),
                      "embedded": STATE["vectors"] is not None, "indexed": bool(STATE["index"])}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read(); name = file.filename or "upload"
    if name.lower().endswith(".pdf"):
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for pg in pdf.pages:
                    pages.append(re.sub(r"Page \d+ of \d+", "", pg.extract_text() or "").strip())
            pages = pages or [""]
        except Exception as e:
            raise HTTPException(400, f"PDF parse failed: {e}")
    else:
        pages = paginate(data.decode("utf-8", "ignore"))
    return set_pages(pages, name)

@app.post("/api/upload_text")
def upload_text(req: TextUpload):
    return set_pages(paginate(req.text), req.filename)


# ---------------- chunk (shared, fixed-size) ----------------
@app.post("/api/chunk")
def chunk(req: ChunkReq):
    full = STATE["full"]
    if not full: raise HTTPException(409, "upload a document first")
    size = max(50, req.size); overlap = max(0, min(req.overlap, size - 1)); step = max(1, size - overlap)
    chunks, n = [], 0
    for idx, a in enumerate(range(0, max(1, len(full)), step)):
        b = min(a + size, len(full)); text = full[a:b].strip()
        if text:
            chunks.append({"id": f"c{n+1}", "page_start": page_of(a), "page_end": page_of(max(a, b - 1)),
                           "chars": len(text), "text": text,
                           "overlap_head": overlap if idx > 0 else 0,
                           "overlap_tail": overlap if b < len(full) else 0})
            n += 1
        if b >= len(full): break
    STATE.update(chunks=chunks, vectors=None, model=None, pca_mean=None, pca_comps=None,
                 index={}, df={}, avgdl=0.0)
    return {"count": len(chunks), "size": size, "overlap": overlap, "step": step, "chunks": chunks}


# ---------------- dense: embed ----------------
@app.post("/api/embed")
def embed(req: EmbedReq):
    if not STATE["chunks"]: raise HTTPException(409, "chunk the document first")
    texts = [c["text"] for c in STATE["chunks"]]
    try:
        raw = np.array(llm.embed_texts(texts, model=req.model, sdk=req.sdk), dtype=np.float32)
    except Exception as e:
        raise HTTPException(500, f"embedding failed: {e}")
    dim = raw.shape[1]
    norms = np.linalg.norm(raw, axis=1, keepdims=True); norms[norms == 0] = 1.0
    STATE.update(vectors=raw / norms, model=req.model)

    sim = STATE["vectors"] @ STATE["vectors"].T
    labels = [{"id": c["id"], "page": c["page_start"]} for c in STATE["chunks"]]
    top_pair = None
    if len(STATE["chunks"]) >= 2:
        m = sim.copy(); np.fill_diagonal(m, -2.0)
        i, j = np.unravel_index(np.argmax(m), m.shape)
        top_pair = {"i": int(i), "j": int(j), "score": round(float(sim[i, j]), 3)}
    sim_list = [[round(float(v), 3) for v in row] for row in sim]

    mean = raw.mean(axis=0); centered = raw - mean
    if len(STATE["chunks"]) >= 2:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False); comps = Vt[:2]
    else:
        comps = np.eye(2, dim, dtype=np.float32)
    coords = centered @ comps.T
    STATE.update(pca_mean=mean, pca_comps=comps)
    cmap = [{"id": c["id"], "page": c["page_start"], "x": round(float(coords[k, 0]), 4),
             "y": round(float(coords[k, 1]), 4)} for k, c in enumerate(STATE["chunks"])]

    pv = min(PREVIEW_DIMS, dim)
    scale = float(np.percentile(np.abs(raw[:, :pv]), 98)) or 0.1
    rows = [{"id": c["id"], "page": c["page_start"], "cells": [round(float(v), 4) for v in raw[k, :pv]]}
            for k, c in enumerate(STATE["chunks"])]
    return {"n_chunks": len(STATE["chunks"]), "dim": dim, "sdk": req.sdk, "scale": round(scale, 4),
            "preview_dims": pv, "rows": rows, "map": cmap, "sim": sim_list,
            "labels": labels, "top_pair": top_pair}


# ---------------- sparse: index ----------------
@app.post("/api/index")
def build_index(req: IndexReq):
    if not STATE["chunks"]: raise HTTPException(409, "chunk the document first")
    STATE["stop"], STATE["stem"] = req.stop, req.stem
    index: Dict[str, Dict[str, int]] = {}; dls = []
    for c in STATE["chunks"]:
        terms = analyze(c["text"], req.stop, req.stem)
        tf: Dict[str, int] = {}
        for t in terms: tf[t] = tf.get(t, 0) + 1
        c["_tf"], c["_dl"] = tf, len(terms); dls.append(len(terms))
        for t, n in tf.items(): index.setdefault(t, {})[c["id"]] = n
    df = {t: len(p) for t, p in index.items()}
    STATE.update(index=index, df=df, avgdl=(sum(dls) / len(dls) if dls else 0.0))

    sample = next((c for c in STATE["chunks"] if c["id"] == req.sample_id), STATE["chunks"][0])
    pipe = analyze_pipeline(sample["text"], req.stop, req.stem); pipe["id"] = sample["id"]
    idx_rows = [{"term": t, "df": df[t],
                 "postings": [{"id": cid, "tf": n} for cid, n in sorted(index[t].items(), key=lambda kv: (-kv[1], kv[0]))]}
                for t in sorted(index.keys())]
    return {"n_chunks": len(STATE["chunks"]), "n_terms": len(index), "avgdl": round(STATE["avgdl"], 1),
            "stop": req.stop, "stem": req.stem, "pipeline": pipe, "index": idx_rows}


# ---------------- search: dense + sparse + RRF ----------------
def _idf(term):
    N = len(STATE["chunks"]); df = STATE["df"].get(term, 0)
    return math.log(1 + (N - df + 0.5) / (df + 0.5))

def _bm25_rows(qterms, c):
    rows, total = [], 0.0
    dl = c.get("_dl", 0); avgdl = STATE["avgdl"] or 1.0
    for t in qterms:
        in_vocab = t in STATE["index"]; tf = c.get("_tf", {}).get(t, 0)
        if in_vocab and tf > 0:
            idf = _idf(t); contrib = idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avgdl))
        else:
            contrib = 0.0
        total += contrib
        rows.append({"term": t, "in_vocab": in_vocab, "df": STATE["df"].get(t, 0),
                     "idf": round(_idf(t), 3) if in_vocab else 0.0, "tf": tf, "contrib": round(contrib, 3)})
    return rows, total

@app.post("/api/search")
def search(req: SearchReq):
    if STATE["vectors"] is None: raise HTTPException(409, "embed the chunks first")
    if not STATE["index"]: raise HTTPException(409, "build the index first")
    if not req.query.strip(): raise HTTPException(400, "empty query")
    chunks = STATE["chunks"]
    rk = max(1, min(req.retrieve_k, len(chunks)))   # candidates per lane
    rrf_k = max(1, req.rrf_k)
    tn = max(1, req.top_n)

    # --- dense (cosine over all chunks) ---
    try:
        q_raw = np.array(llm.embed_text(req.query, model=STATE["model"], sdk=req.sdk), dtype=np.float32)
    except Exception as e:
        raise HTTPException(500, f"query embed failed: {e}")
    qn = q_raw / (np.linalg.norm(q_raw) or 1.0)
    cos = STATE["vectors"] @ qn
    dense_order = list(np.argsort(cos)[::-1])
    dense_rank = {int(i): r + 1 for r, i in enumerate(dense_order)}
    dense_results = [{"id": chunks[int(i)]["id"], "page": chunks[int(i)]["page_start"], "rank": r + 1,
                      "score": round(float(cos[int(i)]), 4), "text": chunks[int(i)]["text"]}
                     for r, i in enumerate(dense_order[:rk])]
    query_xy = None
    if STATE["pca_comps"] is not None:
        c2 = (q_raw - STATE["pca_mean"]) @ STATE["pca_comps"].T
        query_xy = {"x": round(float(c2[0]), 4), "y": round(float(c2[1]), 4)}

    # --- sparse (BM25 over all chunks) ---
    qterms_all = analyze(req.query, STATE["stop"], STATE["stem"])
    seen = set(); qterms = [t for t in qterms_all if not (t in seen or seen.add(t))]
    q_status = [{"term": t, "in_vocab": t in STATE["index"], "df": STATE["df"].get(t, 0)} for t in qterms]
    bm = []
    for k, c in enumerate(chunks):
        rows, total = _bm25_rows(qterms, c); bm.append((k, total, rows))
    bm_by_k = {k: tot for k, tot, _ in bm}
    sparse_sorted = sorted(bm, key=lambda x: x[1], reverse=True)
    sparse_rank = {k: r + 1 for r, (k, _, _) in enumerate(sparse_sorted)}
    sparse_results = [{"id": chunks[k]["id"], "page": chunks[k]["page_start"], "rank": r + 1,
                       "score": round(tot, 3), "text": chunks[k]["text"], "terms": rows}
                      for r, (k, tot, rows) in enumerate(sparse_sorted[:rk])]

    # --- hybrid (RRF). Only candidates inside either lane's retrieve_k get a contribution. ---
    dense_topset = {int(i) for i in dense_order[:rk]}
    sparse_topset = {k for k, _, _ in sparse_sorted[:rk]}
    pool = dense_topset | sparse_topset
    fused = []
    for k in pool:
        dr, sr = dense_rank[k], sparse_rank[k]
        rrf = (1.0 / (rrf_k + dr) if k in dense_topset else 0.0) + \
              (1.0 / (rrf_k + sr) if k in sparse_topset else 0.0)
        fused.append((k, rrf, dr, sr))
    fused.sort(key=lambda x: x[1], reverse=True)
    rrf_rank = {k: r + 1 for r, (k, _, _, _) in enumerate(fused)}
    hybrid_results = [{"id": chunks[k]["id"], "page": chunks[k]["page_start"], "rank": rrf_rank[k],
                       "rrf": round(rrf, 5), "dense_rank": dr, "sparse_rank": sr,
                       "dense_score": round(float(cos[k]), 4), "sparse_score": round(bm_by_k[k], 3),
                       "in_dense": k in dense_topset, "in_sparse": k in sparse_topset,
                       "text": chunks[k]["text"]} for k, rrf, dr, sr in fused[:tn]]

    return {"retrieve_k": rk, "rrf_k": rrf_k, "top_n": tn, "n_chunks": len(chunks),
            "dense": {"results": dense_results, "query_xy": query_xy},
            "sparse": {"q_terms": qterms, "q_status": q_status, "results": sparse_results},
            "hybrid": {"results": hybrid_results, "rrf_k": rrf_k}}


# ---------------- rerank (cross-encoder-style LLM judge) ----------------
@app.post("/api/rerank")
def rerank(req: RerankReq):
    if not req.candidates: raise HTTPException(400, "no candidates to rerank")
    by_id = {c["id"]: c for c in STATE["chunks"]}
    items = [{"id": c["id"], "text": by_id[c["id"]]["text"]} for c in req.candidates if c["id"] in by_id]
    try:
        scores = llm.rerank_scores(req.query, items, model=req.model, sdk=req.sdk)
    except Exception as e:
        raise HTTPException(500, f"rerank failed: {e}")
    out = []
    for c in req.candidates:
        sc = scores.get(c["id"])
        out.append({**c, "page": by_id[c["id"]]["page_start"], "text": by_id[c["id"]]["text"],
                    "judge_score": (round(float(sc), 2) if sc is not None else None)})
    # rank by judge score (fall back to existing RRF rank if the judge returned nothing)
    if scores:
        out.sort(key=lambda x: (x["judge_score"] if x["judge_score"] is not None else -1), reverse=True)
    else:
        out.sort(key=lambda x: x.get("rrf_rank", 999))
    for r, o in enumerate(out): o["rerank_rank"] = r + 1
    return {"results": out[:max(1, req.top_n)], "used_judge": bool(scores)}


# ---------------- answer ----------------
@app.post("/api/answer")
def answer(req: AnswerReq):
    if req.chunk_ids:
        by_id = {c["id"]: c for c in STATE["chunks"]}
        hits = [{"id": cid, "page": by_id[cid]["page_start"], "text": by_id[cid]["text"]}
                for cid in req.chunk_ids if cid in by_id]
    else:
        s = search(SearchReq(query=req.question, retrieve_k=req.retrieve_k, rrf_k=req.rrf_k,
                             top_n=req.top_n, sdk=req.sdk))
        hits = [{"id": h["id"], "page": h["page"], "text": h["text"]} for h in s["hybrid"]["results"]]
    if not hits: raise HTTPException(409, "no chunks to answer from")
    try:
        ans = llm.answer(req.question, hits, model=req.model, sdk=req.sdk)
    except Exception as e:
        raise HTTPException(500, f"answer failed: {e}")
    return {"answer": ans, "matches": hits}
