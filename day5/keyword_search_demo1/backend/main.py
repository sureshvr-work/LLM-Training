"""
main.py  —  the Keyword Lab backend, one endpoint per stage of the SPARSE pipeline.

The dense lab embedded chunks into vectors and searched by cosine. This lab does
the opposite: it analyzes chunks into terms, files them in an inverted index, and
searches by BM25. There is no embedding model anywhere — the only LLM call is the
final answer (and, if you pick it, the LLM-driven chunker).

One document at a time, state in a single in-memory dict (a teaching demo for one
user — no database, no sessions). Stages, in UI order:
    /api/upload      -> parse a PDF (or text) into per-page text
    /api/strategies  -> the chunking strategies that apply to an inverted index
    /api/chunk       -> cut the document with the chosen strategy
    /api/index       -> analyzer (normalize/tokenize/stop/stem) -> term -> postings
    /api/search      -> analyze the query the SAME way, BM25 vs every chunk, top-K
    /api/answer      -> hand the retrieved chunks to the LLM, return the answer
"""
import io
import re

import pdfplumber
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import engine as E
from llm import answer as llm_answer, segment as llm_segment

app = FastAPI(title="Keyword Lab")

# ---- single-document state, in memory --------------------------------------
STATE = {
    "full": "",          # whole document as one string
    "starts": [],        # char offset where each page begins
    "source": None,      # filename (travels with every chunk as metadata)
    "chunks": [],        # finalized chunk dicts (the searchable units)
    "strategy": None,
    "index": None,       # term -> {df, postings}
    "doc_len": {},       # chunk_id -> token count
    "avgdl": 0.0,
    "analyzer": {"stop": True, "stem": True},
}


def _ingest(pages, filename):
    """Common path for PDF and text: join pages, remember page offsets."""
    full, starts = "", []
    for i, text in enumerate(pages):
        starts.append(len(full))
        full += text
        if i < len(pages) - 1:
            full += "\n"
    STATE.update(full=full, starts=starts, source=filename,
                 chunks=[], strategy=None, index=None, doc_len={}, avgdl=0.0)
    return {
        "filename": filename,
        "pages": len(pages),
        "total_chars": len(full),
        "per_page": [{"page": i + 1, "chars": len(t)} for i, t in enumerate(pages)],
    }


# ---- STAGE 1: upload (PDF or text) -----------------------------------------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    name = file.filename or "upload"
    if name.lower().endswith(".pdf"):
        pages = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                lines = [ln for ln in text.split("\n")
                         if not re.search(r"Page\s+\d+\s+of\s+\d+\s*$", ln.strip())]
                pages.append("\n".join(lines))
    else:
        text = raw.decode("utf-8", errors="replace")
        # Treat form-feeds as page breaks; otherwise ~1800 chars per "page" so
        # the per-page bar chart and page badges still mean something.
        if "\f" in text:
            pages = text.split("\f")
        else:
            pages = [text[i:i + 1800] for i in range(0, len(text), 1800)] or [text]
    return _ingest(pages, name)


class TextRequest(BaseModel):
    text: str
    filename: str = "pasted.txt"


@app.post("/api/upload_text")
def upload_text(req: TextRequest):
    text = req.text
    pages = text.split("\f") if "\f" in text else [text[i:i + 1800] for i in range(0, len(text), 1800)] or [text]
    return _ingest(pages, req.filename)


# ---- the strategy catalogue (what applies to an inverted index) ------------
@app.get("/api/strategies")
def strategies():
    return {"strategies": E.STRATEGIES,
            "excluded": {"family": "Embedding-time",
                         "examples": "late chunking, ColBERT",
                         "why": "lives in vector space — there is no inverted-index form of it"}}


# ---- STAGE 2: chunk with the chosen strategy -------------------------------
class ChunkRequest(BaseModel):
    strategy: str = "fixed"
    params: dict = {}
    sdk: str = "openai"          # only used by the LLM-driven strategy
    model: str = "gpt-4o-mini"


@app.post("/api/chunk")
def chunk(req: ChunkRequest):
    full = STATE["full"]
    page_of = E.make_page_of(STATE["starts"])
    p = req.params or {}
    strat = req.strategy
    meta = {}

    if strat == "fixed":
        spans, meta = E.chunk_fixed(full, int(p.get("size", 700)), int(p.get("overlap", 120)))
    elif strat == "structure":
        spans, meta = E.chunk_structure(full, int(p.get("size", 700)))
    elif strat == "meaning":
        spans, meta = E.chunk_meaning(full, int(p.get("size", 650)))
    elif strat == "hierarchical":
        spans, meta = E.chunk_hierarchical(full, int(p.get("parent", 1500)), int(p.get("child", 350)))
    elif strat == "enrichment":
        title = (STATE["source"] or "document").rsplit(".", 1)[0]
        spans, meta = E.chunk_enrichment(full, int(p.get("size", 600)), doc_title=title)
    elif strat == "llm":
        sections = llm_segment(full, model=req.model, sdk=req.sdk)
        spans = E.spans_from_sections(full, sections)
        meta = {"sections": len(sections), "via": req.sdk}
    else:
        return {"error": f"unknown strategy {strat!r}"}

    chunks = E.finalize_chunks(spans, full, page_of)
    STATE.update(chunks=chunks, strategy=strat, index=None, doc_len={}, avgdl=0.0)

    desc = next((s for s in E.STRATEGIES if s["id"] == strat), {})
    return {
        "strategy": strat, "name": desc.get("name", strat), "family": desc.get("family", ""),
        "count": len(chunks), "meta": meta,
        "chunks": [{
            "id": c["id"], "page_start": c["page_start"], "page_end": c["page_end"],
            "chars": c["chars"], "source": STATE["source"],
            "body": c["body"], "text": c["text"],
            "context": c.get("context"),
            "overlap_head": c.get("overlap_head", 0), "overlap_tail": c.get("overlap_tail", 0),
            "parent_id": c.get("parent_id"),
        } for c in chunks],
    }


# ---- STAGE 3: build the inverted index (the analyzer) ----------------------
class IndexRequest(BaseModel):
    stop: bool = True
    stem: bool = True
    sample_id: str = None       # which chunk to show the analyzer pipeline for


@app.post("/api/index")
def index(req: IndexRequest):
    chunks = STATE["chunks"]
    idx, dl, avgdl = E.build_index(chunks, stop=req.stop, do_stem=req.stem)
    STATE.update(index=idx, doc_len=dl, avgdl=avgdl, analyzer={"stop": req.stop, "stem": req.stem})

    # analyzer pipeline preview on one chunk (its indexed text = context+body)
    sample = next((c for c in chunks if c["id"] == req.sample_id), chunks[0] if chunks else None)
    pipeline = None
    if sample:
        st = E.analyze_steps(sample["text"], stop=req.stop, do_stem=req.stem)
        pipeline = {"id": sample["id"], **st}

    # inverted index, sorted by df desc then term — show the whole thing (small corpus)
    terms = sorted(idx.items(), key=lambda kv: (-kv[1]["df"], kv[0]))
    index_rows = [{"term": t, "df": e["df"],
                   "postings": [{"id": cid, "tf": tf} for cid, tf in
                                sorted(e["postings"].items())]}
                  for t, e in terms]
    return {
        "n_chunks": len(chunks), "n_terms": len(idx), "avgdl": round(avgdl, 1),
        "stop": req.stop, "stem": req.stem,
        "pipeline": pipeline, "index": index_rows,
        "doc_len": [{"id": c["id"], "len": dl.get(c["id"], 0)} for c in chunks],
    }


# ---- STAGE 4: search (BM25) ------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 4


@app.post("/api/search")
def search(req: SearchRequest):
    chunks = STATE["chunks"]
    by_id = {c["id"]: c for c in chunks}
    a = STATE["analyzer"]
    r = E.search(req.query, chunks, STATE["index"], STATE["doc_len"], STATE["avgdl"],
                 top_k=req.top_k, stop=a["stop"], do_stem=a["stem"])

    def pack(row):
        c = by_id[row["id"]]
        return {
            "id": c["id"], "page": c["page_start"], "source": STATE["source"],
            "score": row["score"], "terms": row["terms"],
            "text": c["body"].strip(), "context": c.get("context"),
            "parent_id": c.get("parent_id"),
        }

    return {
        "query": req.query, "top_k": req.top_k,
        "q_terms": r["q_terms"], "q_status": r["q_status"],
        "results": [pack(x) for x in r["top"]],
        "all_scored": [{"id": x["id"], "score": x["score"]} for x in r["ranked"]],
    }


# ---- STAGE 5: answer (LLM) -------------------------------------------------
class AnswerRequest(BaseModel):
    question: str
    top_k: int = 4
    sdk: str = "openai"
    model: str = "gpt-4o-mini"


@app.post("/api/answer")
def answer_endpoint(req: AnswerRequest):
    chunks = STATE["chunks"]
    by_id = {c["id"]: c for c in chunks}
    a = STATE["analyzer"]
    r = E.search(req.question, chunks, STATE["index"], STATE["doc_len"], STATE["avgdl"],
                 top_k=req.top_k, stop=a["stop"], do_stem=a["stem"])

    matches, context_items = [], []
    for row in r["top"]:
        c = by_id[row["id"]]
        # Hierarchical: the child is indexed, but the PARENT is what the LLM reads.
        ctx_text = c.get("parent_text") or c["body"]
        context_items.append({"id": c["id"], "page": c["page_start"],
                              "source": STATE["source"], "text": ctx_text})
        matches.append({"id": c["id"], "page": c["page_start"], "source": STATE["source"],
                        "score": row["score"], "text": c["body"].strip(),
                        "parent_id": c.get("parent_id")})

    out = llm_answer(req.question, context_items, model=req.model, sdk=req.sdk)
    return {"answer": out, "sdk_used": req.sdk, "model": req.model, "matches": matches}


@app.get("/api/health")
def health():
    return {"status": "ok", "chars": len(STATE["full"]),
            "chunks": len(STATE["chunks"]), "indexed": STATE["index"] is not None}
