"""
engine.py  —  the keyword (sparse) retrieval engine, with NO external deps.

This is the heart of the lab and the deliberate contrast to the dense demo:
there is NO embedding model and NO API call anywhere in this file. Keyword
search lives entirely in the realm of exact strings — analyze text into terms,
file each term under the chunks that contain it (the inverted index), then
score overlap with BM25. That absence is the lesson.

It is split out from main.py on purpose: everything here is pure Python, so it
can be unit-tested without FastAPI, pdfplumber, or an OpenAI key.

Pipeline mirrored by the UI:
    chunk(strategy)  -> cut the document (6 strategies that all apply to keyword)
    build_index()    -> analyzer (normalize/tokenize/stop/stem) -> term -> postings
    search(query)    -> analyze the query with the SAME analyzer, BM25 vs every chunk
"""
import re

# ============================================================================
# 1. THE ANALYZER  —  raw text  ->  index terms
# ============================================================================
# A keyword index can only match strings, so every chunk (and every query) is
# reduced to a bag of normalized TERMS. The query MUST run through the very same
# analyzer as the documents — if the index lowercases but the query doesn't,
# "AML" and "aml" become different strings and you silently get zero hits. That
# shared-transform rule is the keyword equivalent of "embed the query with the
# same model you embedded the chunks with."

# A compact, honest stopword list (function words that carry no retrieval signal).
STOPWORDS = set((
    "a an and are as at be been being by for from had has have he her his in into is it its "
    "of on or that the their them they this to was were will with within would you your "
    "but not no any can could should may might must each other than then there these those "
    "we our us i if so such over under out up down per via about above below between"
).split())

# Keep alphanumerics together; allow internal - / . so codes survive whole
# (e.g. 0078-0357-15, kyc/aml, fincen.gov). Everything else is a separator.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")
# Join thousands separators so "$10,000" -> token "10000" (a real index quirk worth showing).
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d)")


def char_filter(text: str) -> str:
    """Normalization step: lowercase + glue digit groups (10,000 -> 10000)."""
    return _THOUSANDS_RE.sub("", text.lower())


def stem(t: str) -> str:
    """
    A deliberately tiny suffix stemmer. Its whole pedagogical job is to show
    that stemming folds word FORMS (reports -> report, monitoring -> monitor)
    but never word MEANINGS (wire never becomes transfer). Codes/numbers and
    very short tokens are left untouched.
    """
    if any(ch.isdigit() for ch in t) or len(t) <= 3:
        return t
    for suf in ("ization", "iveness", "ation", " tion", "tion", "sion",
                "ing", "edly", "ed", "ly", "es", "s"):
        suf = suf.strip()
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: len(t) - len(suf)]
    return t


def analyze(text: str, stop: bool = True, do_stem: bool = True):
    """raw text -> list of index terms (in order, duplicates kept for tf)."""
    toks = _TOKEN_RE.findall(char_filter(text))
    if stop:
        toks = [t for t in toks if t not in STOPWORDS]
    if do_stem:
        toks = [stem(t) for t in toks]
    return toks


def analyze_steps(text: str, stop: bool = True, do_stem: bool = True):
    """
    Same as analyze(), but returns each pipeline stage so the UI can show the
    funnel: normalized tokens -> after stopwords -> after stemming.
    """
    lowered = _TOKEN_RE.findall(char_filter(text))
    after_stop = [t for t in lowered if not (stop and t in STOPWORDS)]
    final = [stem(t) if do_stem else t for t in after_stop]
    return {
        "lowered": lowered,
        "after_stop": after_stop,
        "final": final,
        "stopped": [t for t in lowered if stop and t in STOPWORDS],
    }


# ============================================================================
# 2. THE INVERTED INDEX  +  BM25
# ============================================================================
# index[term] = {"df": int, "postings": {chunk_id: term_frequency}}
# Plus per-chunk token length and the corpus average, both needed by BM25's
# length normalization.
K1, B = 1.5, 0.75


def build_index(chunks, stop=True, do_stem=True):
    """
    chunks: list of dicts that each have 'id' and 'text' (the SEARCHABLE text;
            for enrichment that already includes the prepended context).
    Returns (index, doc_len, avgdl).
    """
    index = {}
    doc_len = {}
    total = 0
    for c in chunks:
        toks = analyze(c["text"], stop=stop, do_stem=do_stem)
        doc_len[c["id"]] = len(toks)
        total += len(toks)
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t, f in tf.items():
            e = index.setdefault(t, {"df": 0, "postings": {}})
            e["df"] += 1
            e["postings"][c["id"]] = f
    avgdl = (total / len(chunks)) if chunks else 0.0
    return index, doc_len, avgdl


def _idf(df, n):
    # Non-negative BM25 idf. Rare term (small df) -> large idf -> counts more.
    import math
    return math.log(1 + (n - df + 0.5) / (df + 0.5))


def score_chunk(q_terms, chunk_id, index, doc_len, avgdl, n):
    """
    BM25 score of one chunk for the analyzed query terms, with a per-term
    breakdown so the UI can show WHY a chunk scored what it did. A query term
    that is absent from the chunk contributes exactly 0 — the mechanical reason
    a vocabulary mismatch yields a zero score.
    """
    dl = doc_len.get(chunk_id, 0)
    score = 0.0
    breakdown = []
    for t in q_terms:
        e = index.get(t)
        if not e:  # term in NO chunk at all -> df 0, can never contribute
            breakdown.append({"term": t, "in_vocab": False, "df": 0,
                              "idf": 0.0, "tf": 0, "contrib": 0.0})
            continue
        tf = e["postings"].get(chunk_id, 0)
        idf = _idf(e["df"], n)
        if tf == 0:
            contrib = 0.0
        else:
            denom = tf + K1 * (1 - B + B * (dl / avgdl if avgdl else 1))
            contrib = idf * (tf * (K1 + 1)) / denom
        score += contrib
        breakdown.append({"term": t, "in_vocab": True, "df": e["df"],
                          "idf": round(idf, 3), "tf": tf, "contrib": round(contrib, 4)})
    return score, breakdown


def search(query, chunks, index, doc_len, avgdl, top_k=4, stop=True, do_stem=True):
    """Analyze the query with the SAME analyzer, BM25 vs every chunk, rank."""
    n = len(chunks)
    q_terms = analyze(query, stop=stop, do_stem=do_stem)
    ranked = []
    for c in chunks:
        s, bd = score_chunk(q_terms, c["id"], index, doc_len, avgdl, n)
        ranked.append({"id": c["id"], "score": round(s, 4), "terms": bd})
    ranked.sort(key=lambda r: r["score"], reverse=True)
    # which query terms exist anywhere in the corpus (for the live echo)
    q_status = [{"term": t, "in_vocab": t in index, "df": index.get(t, {}).get("df", 0)}
                for t in q_terms]
    return {"q_terms": q_terms, "q_status": q_status,
            "ranked": ranked, "top": ranked[:top_k]}


# ============================================================================
# 3. PAGE MAPPING HELPER
# ============================================================================
import bisect


def make_page_of(starts):
    """Return a fn: char offset -> 1-indexed page, given page start offsets."""
    def page_of(ix):
        return bisect.bisect_right(starts, ix)
    return page_of


# ============================================================================
# 4. THE SIX CHUNKING STRATEGIES THAT APPLY TO THE INVERTED INDEX
# ============================================================================
# The CUT is engine-agnostic: the same chunk you would embed, you also index.
# Six of the seven taxonomy families produce ordinary text chunks that index
# cleanly. (The seventh — Embedding-time: late chunking / ColBERT — is vector
# only and has no inverted-index form, so it is absent here by design.)
#
# Every chunker returns a list of "span" dicts with at least start/end offsets
# into `full`; finalize_chunks() then stamps ids, pages, and char counts. Two
# strategies attach extra fields:
#   enrichment   -> "context"     (prepended text that also gets indexed)
#   hierarchical -> "parent_id" + "parent_text" (child is indexed, parent returned)

STRATEGIES = [
    {"id": "fixed", "family": "Fixed / rule-based",
     "name": "Fixed-size window",
     "blurb": "Cut a fixed character window, slide forward, repeat. Overlap keeps facts from being split across a boundary.",
     "params": [{"key": "size", "label": "size (chars)", "min": 200, "max": 1500, "step": 50, "default": 700},
                {"key": "overlap", "label": "overlap (chars)", "min": 0, "max": 400, "step": 10, "default": 120}]},
    {"id": "structure", "family": "Structure-aware",
     "name": "Structure-aware (recursive)",
     "blurb": "Split on the document's own seams — blank lines, then lines, then sentences — never mid-paragraph if it can help it.",
     "params": [{"key": "size", "label": "target (chars)", "min": 300, "max": 1500, "step": 50, "default": 700}]},
    {"id": "meaning", "family": "Meaning-based",
     "name": "Meaning-based (lexical cohesion)",
     "blurb": "Group adjacent sentences while they share vocabulary; cut where the wording shifts. A no-model proxy for semantic chunking (TextTiling-style).",
     "params": [{"key": "size", "label": "target (chars)", "min": 300, "max": 1500, "step": 50, "default": 650}]},
    {"id": "hierarchical", "family": "Hierarchical",
     "name": "Hierarchical (small-to-big)",
     "blurb": "Index small CHILD chunks for precise matching, but return the larger PARENT for full context. Lane-agnostic retrieval pattern.",
     "params": [{"key": "parent", "label": "parent (chars)", "min": 800, "max": 2400, "step": 100, "default": 1500},
                {"key": "child", "label": "child (chars)", "min": 150, "max": 600, "step": 25, "default": 350}]},
    {"id": "enrichment", "family": "Enrichment",
     "name": "Enrichment (contextual headers)",
     "blurb": "Prepend each chunk with its document + nearest heading before indexing, so a chunk that says only '$10,000' still answers 'AML threshold'. Boosts BM25, not just embeddings.",
     "params": [{"key": "size", "label": "size (chars)", "min": 200, "max": 1200, "step": 50, "default": 600}]},
    {"id": "llm", "family": "LLM-driven",
     "name": "LLM-driven segmentation",
     "blurb": "Ask the model to split the document into titled sections; each section becomes a chunk (its title is indexed too). Costs one API call.",
     "params": []},
]


def finalize_chunks(spans, full, page_of):
    """Stamp ids/pages/chars onto raw spans, computing fixed-style overlaps."""
    n = len(spans)
    out = []
    for i, sp in enumerate(spans):
        s, e = sp["start"], sp["end"]
        body = full[s:e]
        context = sp.get("context")
        # searchable text = (context + body) for enrichment, else body
        searchable = (context + "\n" + body) if context else body
        head = (spans[i - 1]["end"] - s) if i > 0 else 0
        tail = (e - spans[i + 1]["start"]) if i < n - 1 else 0
        c = {
            "id": f"c{i+1}",
            "text": searchable,
            "body": body,
            "page_start": page_of(s),
            "page_end": page_of(max(s, e - 1)),
            "chars": len(body),
            "overlap_head": max(0, head),
            "overlap_tail": max(0, tail),
        }
        if context:
            c["context"] = context
        if "parent_id" in sp:
            c["parent_id"] = sp["parent_id"]
            c["parent_text"] = sp["parent_text"]
        out.append(c)
    return out


# ---- 4a. fixed ------------------------------------------------------------
def chunk_fixed(full, size=700, overlap=120):
    size = max(50, size)
    overlap = min(max(0, overlap), size - 1)
    step = size - overlap
    spans, start = [], 0
    while start < len(full):
        end = min(start + size, len(full))
        spans.append({"start": start, "end": end})
        if end >= len(full):
            break
        start += step
    return spans, {"size": size, "overlap": overlap, "step": step}


# ---- 4b. structure-aware (recursive) --------------------------------------
def _recursive_spans(full, lo, hi, seps, target):
    if hi - lo <= target or not seps:
        return [(lo, hi)]
    sep = seps[0]
    pieces, i = [], lo
    while True:
        j = full.find(sep, i, hi)
        if j < 0:
            break
        pieces.append((i, j + len(sep)))
        i = j + len(sep)
    pieces.append((i, hi))
    if len(pieces) == 1:  # separator absent here -> descend
        return _recursive_spans(full, lo, hi, seps[1:], target)
    out, curlo, curhi = [], None, None
    for a, b in pieces:
        if b - a > target:
            if curlo is not None:
                out.append((curlo, curhi)); curlo = None
            out += _recursive_spans(full, a, b, seps[1:], target)
        elif curlo is None:
            curlo, curhi = a, b
        elif b - curlo <= target:
            curhi = b
        else:
            out.append((curlo, curhi)); curlo, curhi = a, b
    if curlo is not None:
        out.append((curlo, curhi))
    return out


def chunk_structure(full, size=700):
    spans = _recursive_spans(full, 0, len(full), ["\n\n", "\n", ". ", " "], size)
    return [{"start": s, "end": e} for s, e in spans if e > s], {"size": size}


# ---- 4c. meaning-based (lexical cohesion, no model) -----------------------
_SENT_RE = re.compile(r"[^.!?]*[.!?]+(?:\s+|$)|[^.!?]+$")


def _sentences(full):
    out, pos = [], 0
    for m in _SENT_RE.finditer(full):
        s, e = m.start(), m.end()
        if e > s and full[s:e].strip():
            out.append((s, e))
        pos = e
    return out or [(0, len(full))]


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def chunk_meaning(full, size=650):
    """
    Group adjacent sentences while their vocabulary overlaps; cut at the low
    points (topic shifts). A purely lexical proxy for semantic chunking — no
    embedding model — which is exactly why it belongs in a keyword demo.
    """
    sents = _sentences(full)
    toks = [analyze(full[s:e]) for s, e in sents]
    spans, cur_lo, cur_hi, last_i = [], sents[0][0], sents[0][1], 0
    for i in range(1, len(sents)):
        s, e = sents[i]
        sim = _jaccard(toks[i - 1], toks[i])
        cur_size = cur_hi - cur_lo
        # cut if we're past target AND the wording just shifted, or hard cap at 1.7x
        if (cur_size >= size and sim < 0.18) or cur_size >= size * 1.7:
            spans.append((cur_lo, cur_hi))
            cur_lo, cur_hi = s, e
        else:
            cur_hi = e
    spans.append((cur_lo, cur_hi))
    return [{"start": s, "end": e} for s, e in spans], {"size": size}


# ---- 4d. hierarchical (small child indexed, big parent returned) ----------
def chunk_hierarchical(full, parent=1500, child=350):
    parent_spans, _ = chunk_structure(full, size=parent)
    spans = []
    for pi, (ps) in enumerate(parent_spans):
        p_lo, p_hi = ps["start"], ps["end"]
        p_text = full[p_lo:p_hi]
        kids, _ = chunk_fixed(full[p_lo:p_hi], size=child, overlap=max(20, child // 8))
        for k in kids:
            spans.append({
                "start": p_lo + k["start"], "end": p_lo + k["end"],
                "parent_id": f"p{pi+1}", "parent_text": p_text,
            })
    return spans, {"parent": parent, "child": child, "parents": len(parent_spans)}


# ---- 4e. enrichment (contextual header prepended, then indexed) -----------
_HEADING_RE = re.compile(r"^\s*(#{1,6}\s+.*|[A-Z0-9][A-Z0-9 /&'-]{3,60})\s*$")


def _looks_like_heading(line):
    """A standalone heading: short, no terminal sentence punctuation, capitalized."""
    if not line or len(line) > 60:
        return False
    if line.startswith("#"):
        return True
    if line[-1] in ".!?,;:":
        return False
    if not line[0].isupper():
        return False
    # at least two words, mostly alphabetic (excludes stray long-ish sentences)
    words = line.split()
    return len(words) >= 1 and line.isupper() or (line[0].isupper() and len(words) <= 9)


def _heading_before(full, pos):
    """Nearest preceding COMPLETE heading line (markdown #, CAPS, or Title Case)."""
    best = None
    for m in re.finditer(r"(?m)^.*$", full):
        if m.start() >= pos:
            break
        if m.end() > pos:           # line is cut by pos -> not a complete heading
            continue
        line = m.group(0).strip()
        if line and _looks_like_heading(line):
            best = line.lstrip("# ").strip()
    return best


def chunk_enrichment(full, size=600, doc_title="document"):
    base, _ = chunk_fixed(full, size=size, overlap=max(40, size // 10))
    for sp in base:
        head = _heading_before(full, sp["start"])
        ctx = f"[{doc_title}" + (f" — {head}]" if head else "]")
        sp["context"] = ctx
    return base, {"size": size}


# ---- 4f. llm-driven: spans are built in main.py from the model's sections --
def spans_from_sections(full, sections):
    """
    sections: list of {"title": str, "text": str} from the LLM. We locate each
    section's text inside `full` to recover real offsets (so page numbers stay
    honest); the title is prepended as indexed context.
    """
    spans, cursor = [], 0
    for sec in sections:
        body = (sec.get("text") or "").strip()
        if not body:
            continue
        probe = body[:40]
        j = full.find(probe, cursor)
        if j < 0:
            j = full.find(probe)
        if j < 0:
            j = cursor
        end = min(len(full), j + len(body))
        title = (sec.get("title") or "").strip()
        sp = {"start": j, "end": end}
        if title:
            sp["context"] = f"[{title}]"
        spans.append(sp)
        cursor = end
    return spans or [{"start": 0, "end": len(full)}]
