"""
main.py — Vectorless RAG demo backend.

Retrieves relevant chunks using BM25 keyword scoring.
No embedding model. No vector DB. No numpy. Pure Python.

Stages (each is an endpoint the UI calls in order):
    /api/upload  -> parse PDF into per-page text
    /api/chunk   -> fixed-size chunking with overlap + page tracking
    /api/index   -> build BM25 index over all chunks (counts terms, IDF weights)
    /api/search  -> score query against every chunk, return ranked top-K + matched terms
    /api/answer  -> hand top-K chunks to the LLM, return cited answer

BM25 (Best Match 25) is the same algorithm used inside Elasticsearch and
Apache Lucene. It improves on raw TF-IDF by:
  - Saturating term frequency (a word appearing 10x is not 10x better than 5x)
  - Normalising for document length (longer docs aren't unfairly rewarded)
"""

import bisect
import io
import math
import re
from collections import Counter, defaultdict

import pdfplumber
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from llm import answer

app = FastAPI(title="Vectorless RAG lab")

# ---------------------------------------------------------------------------
# Single-document in-memory state
# ---------------------------------------------------------------------------
STATE: dict = {
    "pages":  [],
    "full":   "",
    "starts": [],
    "chunks": [],
    "index":  None,   # BM25Index instance
    "source": None,
}

PREVIEW_TERMS = 20   # columns in the term-frequency heatmap


# ---------------------------------------------------------------------------
# BM25 — pure Python, no external dependencies
# ---------------------------------------------------------------------------
_STOP = {
    "a","an","the","is","it","in","on","at","to","of","for","and","or",
    "be","was","are","were","has","have","had","not","with","as","by","from",
    "this","that","these","those","its","their","our","your","my","his","her",
    "i","we","you","he","she","they","do","did","does","can","will","would",
    "could","should","may","might","also","than","then","which","what","when",
    "where","who","how","if","so","but","up","out","no","more","about","into",
    "all","any","each","such","other","both","only","just","over","after",
    "before","between","through","during","without","within","against","per",
}

def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\b[a-z]{2,}\b", text.lower()) if w not in _STOP]


class BM25Index:
    """
    BM25 relevance scoring.

    k1 = 1.5  — term-frequency saturation: diminishing returns after ~3 occurrences
    b  = 0.75 — length normalisation: penalises chunks longer than average
    """
    k1 = 1.5
    b  = 0.75

    def __init__(self, texts: list[str]):
        self.n         = len(texts)
        self.tokenized = [_tokenize(t) for t in texts]
        self.tfs       = [Counter(toks) for toks in self.tokenized]
        self.lengths   = [len(toks) for toks in self.tokenized]
        self.avgdl     = sum(self.lengths) / max(self.n, 1)

        # document frequency: number of chunks containing each term
        self.df: dict[str, int] = defaultdict(int)
        for tf in self.tfs:
            for term in tf:
                self.df[term] += 1

    def idf(self, term: str) -> float:
        """Inverse document frequency — rare terms score higher."""
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def score_all(self, query: str) -> tuple[list[float], list[str]]:
        """Score every chunk against the query. Returns (scores, query_terms)."""
        q_terms = list(dict.fromkeys(_tokenize(query)))  # dedup, preserve order
        scores  = []
        for i, tf in enumerate(self.tfs):
            dl = self.lengths[i]
            s  = 0.0
            for term in q_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                # BM25 formula
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s  += self.idf(term) * num / den
            scores.append(round(s, 4))
        return scores, q_terms

    def top_global_terms(self, n: int) -> list[str]:
        """Top N terms by collection frequency — used as heatmap columns."""
        global_counts: Counter = Counter()
        for tf in self.tfs:
            global_counts.update(tf)
        return [t for t, _ in global_counts.most_common(n)]

    def freq_row(self, chunk_idx: int, terms: list[str]) -> list[int]:
        tf = self.tfs[chunk_idx]
        return [tf.get(t, 0) for t in terms]

    def matched_terms(self, query: str, chunk_idx: int) -> list[str]:
        q_terms = set(_tokenize(query))
        tf      = self.tfs[chunk_idx]
        return [t for t in q_terms if t in tf]


# ---------------------------------------------------------------------------
# Page offset helper
# ---------------------------------------------------------------------------
def _page_of(char_index: int) -> int:
    return bisect.bisect_right(STATE["starts"], char_index)


# ---------------------------------------------------------------------------
# STAGE 1 — Upload & parse PDF
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw   = await file.read()
    pages = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")

    full, starts = "", []
    for i, text in enumerate(pages):
        starts.append(len(full))
        full += text
        if i < len(pages) - 1:
            full += "\n"

    STATE.update(pages=pages, full=full, starts=starts,
                 chunks=[], index=None, source=file.filename)

    return {
        "filename":    file.filename,
        "pages":       len(pages),
        "total_chars": len(full),
        "per_page":    [{"page": i + 1, "chars": len(t)} for i, t in enumerate(pages)],
    }


# ---------------------------------------------------------------------------
# STAGE 2 — Fixed-size chunking with overlap
# ---------------------------------------------------------------------------
class ChunkRequest(BaseModel):
    size:    int = 800
    overlap: int = 150


@app.post("/api/chunk")
def chunk(req: ChunkRequest):
    full    = STATE["full"]
    size    = max(50, req.size)
    overlap = min(max(0, req.overlap), size - 1)
    step    = size - overlap

    chunks, start, i = [], 0, 0
    while start < len(full):
        end = min(start + size, len(full))
        chunks.append({
            "i": i, "id": f"c{i+1}", "text": full[start:end],
            "start": start, "end": end,
            "page_start": _page_of(start),
            "page_end":   _page_of(max(start, end - 1)),
        })
        i += 1
        if end >= len(full):
            break
        start += step

    STATE.update(chunks=chunks, index=None)

    n   = len(chunks)
    out = []
    for idx, c in enumerate(chunks):
        head = (chunks[idx - 1]["end"] - c["start"]) if idx > 0 else 0
        tail = (c["end"] - chunks[idx + 1]["start"]) if idx < n - 1 else 0
        out.append({
            "id":           c["id"],
            "page_start":   c["page_start"],
            "page_end":     c["page_end"],
            "source":       STATE["source"],
            "chars":        len(c["text"]),
            "overlap_head": max(0, head),
            "overlap_tail": max(0, tail),
            "text":         c["text"],
        })

    return {"count": n, "size": size, "overlap": overlap, "step": step, "chunks": out}


# ---------------------------------------------------------------------------
# STAGE 3 — Build BM25 keyword index
# ---------------------------------------------------------------------------
@app.post("/api/index")
def build_index():
    texts = [c["text"] for c in STATE["chunks"]]
    idx   = BM25Index(texts)
    STATE["index"] = idx

    top_terms = idx.top_global_terms(PREVIEW_TERMS)

    rows = []
    for c in STATE["chunks"]:
        ci    = c["i"]
        freqs = idx.freq_row(ci, top_terms)
        top5  = sorted(idx.tfs[ci].items(), key=lambda x: -x[1])[:5]
        rows.append({
            "id":          c["id"],
            "page":        c["page_start"],
            "cells":       freqs,                                          # raw term counts per column
            "top_terms":   [{"term": t, "freq": f} for t, f in top5],    # top 5 for tooltip
            "total_terms": idx.lengths[ci],
        })

    return {
        "n_chunks":          len(texts),
        "vocab_size":        len(idx.df),
        "avg_chunk_length":  round(idx.avgdl),
        "top_terms":         top_terms,   # heatmap column headers
        "rows":              rows,
    }


# ---------------------------------------------------------------------------
# STAGE 4 — BM25 search
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 4


@app.post("/api/search")
def search(req: SearchRequest):
    idx: BM25Index      = STATE["index"]
    scores, q_terms     = idx.score_all(req.query)
    order               = sorted(range(len(scores)), key=lambda i: -scores[i])[:req.top_k]
    max_score           = max((scores[i] for i in order), default=1.0) or 1.0

    results = []
    for rank, ci in enumerate(order, 1):
        c       = STATE["chunks"][ci]
        matched = idx.matched_terms(req.query, ci)
        results.append({
            "rank":             rank,
            "id":               c["id"],
            "page":             c["page_start"],
            "source":           STATE["source"],
            "score":            scores[ci],
            "score_pct":        round(scores[ci] / max_score * 100),   # for bar width
            "matched_terms":    matched,
            "unmatched_terms":  [t for t in q_terms if t not in matched],
            "text":             c["text"].strip(),
        })

    return {
        "query":       req.query,
        "query_terms": q_terms,
        "top_k":       req.top_k,
        "zero_results": all(scores[i] == 0 for i in order),
        "results":     results,
    }


# ---------------------------------------------------------------------------
# STAGE 5 — LLM answer (BM25 retrieval + LLM generation)
# ---------------------------------------------------------------------------
class AnswerRequest(BaseModel):
    question: str
    top_k:    int = 4
    model:    str = "gpt-4o-mini"
    sdk:      str = "openai"


@app.post("/api/answer")
def answer_endpoint(req: AnswerRequest):
    idx: BM25Index  = STATE["index"]
    scores, _       = idx.score_all(req.question)
    order           = sorted(range(len(scores)), key=lambda i: -scores[i])[:req.top_k]

    matches, context_items = [], []
    for rank, ci in enumerate(order, 1):
        c = STATE["chunks"][ci]
        context_items.append({
            "id": c["id"], "page": c["page_start"],
            "source": STATE["source"], "text": c["text"],
        })
        matches.append({
            "rank": rank, "id": c["id"], "page": c["page_start"],
            "source": STATE["source"], "score": round(scores[ci], 4),
            "text": c["text"].strip(),
        })

    llm_answer = answer(req.question, context_items, model=req.model, sdk=req.sdk)
    return {"answer": llm_answer, "model": req.model, "matches": matches}


@app.get("/api/health")
def health():
    return {
        "status":  "ok",
        "pages":   len(STATE["pages"]),
        "chunks":  len(STATE["chunks"]),
        "indexed": STATE["index"] is not None,
    }
