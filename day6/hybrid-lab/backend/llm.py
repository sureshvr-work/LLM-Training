"""Shared OpenAI helpers: embeddings + grounded chat answer.

`sdk="openai"` uses the OpenAI SDK directly. `sdk="langchain"` uses langchain-openai
if it is installed, otherwise it transparently falls back to the OpenAI SDK (same result).
"""
import os
from typing import List, Dict

_client = None
def _openai():
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env file.")
        from openai import OpenAI
        _client = OpenAI()
    return _client


# ---------------- embeddings ----------------
def embed_texts(texts: List[str], model: str = "text-embedding-3-small", sdk: str = "openai") -> List[List[float]]:
    if not texts:
        return []
    if sdk == "langchain":
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model).embed_documents(texts)
        except Exception as e:
            print(f"[llm] langchain embed unavailable ({e}); using OpenAI SDK")
    out: List[List[float]] = []
    cli = _openai()
    B = 96
    for i in range(0, len(texts), B):
        resp = cli.embeddings.create(model=model, input=texts[i:i + B])
        out.extend([d.embedding for d in resp.data])
    return out


def embed_text(text: str, model: str = "text-embedding-3-small", sdk: str = "openai") -> List[float]:
    return embed_texts([text], model=model, sdk=sdk)[0]


# ---------------- grounded answer ----------------
SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the question using ONLY the provided context chunks. "
    "Cite every claim inline with the chunk id and page in square brackets, like [c3, p2]. "
    "If the answer is not in the context, say you don't know. Be concise."
)

def answer(question: str, contexts: List[Dict], model: str = "gpt-4o-mini", sdk: str = "openai") -> str:
    """contexts: [{'id','page','text'}, ...]"""
    ctx = "\n\n".join(f"[{c['id']}, p{c['page']}] {c['text']}" for c in contexts)
    user = f"Question: {question}\n\nContext:\n{ctx}"
    if sdk == "langchain":
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            llm = ChatOpenAI(model=model, temperature=0.2)
            return llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]).content
        except Exception as e:
            print(f"[llm] langchain chat unavailable ({e}); using OpenAI SDK")
    cli = _openai()
    resp = cli.chat.completions.create(
        model=model, temperature=0.2,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


# ---------------- cross-encoder-style reranker (LLM-as-judge) ----------------
# A real cross-encoder reads (query, passage) TOGETHER and scores relevance.
# Here the chat model plays that judge: same idea (joint scoring), no extra model to host.
def rerank_scores(question: str, items: List[Dict], model: str = "gpt-4o-mini", sdk: str = "openai") -> Dict[str, float]:
    """items: [{'id','text'}] -> {id: relevance 0..10}. Falls back to {} on error (caller keeps RRF order)."""
    import json
    listing = "\n".join(f"[{it['id']}] {it['text']}" for it in items)
    sys = ("You are a strict relevance judge. For each passage, score 0-10 how directly it ANSWERS "
           "the question (10 = contains the answer; 0 = unrelated; topical-but-no-answer is around 3-5). "
           'Return ONLY JSON: {"scores": {"<id>": <number>, ...}}.')
    user = f"Question: {question}\n\nPassages:\n{listing}"
    try:
        if sdk == "langchain":
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import SystemMessage, HumanMessage
                txt = ChatOpenAI(model=model, temperature=0).invoke(
                    [SystemMessage(content=sys), HumanMessage(content=user)]).content
                data = json.loads(txt)
            except Exception:
                raise
        else:
            cli = _openai()
            r = cli.chat.completions.create(model=model, temperature=0,
                                            response_format={"type": "json_object"},
                                            messages=[{"role": "system", "content": sys},
                                                      {"role": "user", "content": user}])
            data = json.loads(r.choices[0].message.content)
        return {k: float(v) for k, v in data.get("scores", {}).items()}
    except Exception as e:
        print(f"[llm] rerank failed ({e}); keeping fusion order")
        return {}
