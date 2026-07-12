"""
chat.py  —  the only place that calls an LLM.

Two jobs, both optional to the keyword pipeline:
  answer()   the final RAG step: answer using ONLY the retrieved chunks.
  segment()  the LLM-driven chunker: split a document into titled sections.

Each can run via the OpenAI SDK directly OR via LangChain. Same prompt, same
result — the toggle just shows the two abstraction levels.
"""
import json

from openai import OpenAI

# Lazy client: importing this module never needs the key, only a real call does.
_client = None


def _openai():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


SYSTEM_PROMPT = (
    "You answer questions using ONLY the context passages provided. "
    "Each passage begins with a tag like [c7, p3] meaning chunk id c7 on page 3, "
    "plus its source file. "
    "If the answer is not in the passages, say you don't know. "
    "After each statement, cite the passage tag(s) you used, written exactly as [c7, p3]."
)


def _build_user_prompt(question, context_items):
    blocks = []
    for it in context_items:
        tag = f"[{it['id']}, p{it['page']}]"
        blocks.append(f"{tag} (source: {it['source']})\n{it['text']}")
    context = "\n\n".join(blocks)
    return f"Context passages:\n{context}\n\nQuestion: {question}"


def answer(question, context_items, model="gpt-4o-mini", sdk="openai"):
    """context_items: list of {id, page, source, text}. Returns the answer string."""
    user_prompt = _build_user_prompt(question, context_items)

    if sdk == "langchain":
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = ChatOpenAI(model=model, temperature=0)
        return llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                           HumanMessage(content=user_prompt)]).content

    resp = _openai().chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_prompt}],
    )
    return resp.choices[0].message.content


# ---- LLM-driven chunking ---------------------------------------------------
SEGMENT_SYSTEM = (
    "You split a document into coherent, self-contained sections for retrieval. "
    "Return STRICT JSON: an array of objects with keys \"title\" and \"text\". "
    "The concatenation of every \"text\" must reproduce the document verbatim and "
    "in order — do not paraphrase, summarize, drop, or reorder any text. "
    "Choose section boundaries at natural topic shifts. Output JSON only, no prose."
)


def segment(text, model="gpt-4o-mini", sdk="openai"):
    """Ask the model to split `text` into [{title, text}]. Returns that list."""
    user = f"Split this document into sections.\n\n---\n{text}\n---"

    if sdk == "langchain":
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = ChatOpenAI(model=model, temperature=0)
        raw = llm.invoke([SystemMessage(content=SEGMENT_SYSTEM),
                          HumanMessage(content=user)]).content
    else:
        resp = _openai().chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": SEGMENT_SYSTEM},
                      {"role": "user", "content": user}],
        )
        raw = resp.choices[0].message.content

    raw = raw.strip()
    if raw.startswith("```"):                       # strip ``` / ```json fences
        raw = raw.split("```")[1].lstrip("json").strip() if "```" in raw[3:] else raw.strip("`")
    data = json.loads(raw)
    if isinstance(data, dict):                       # tolerate {"sections":[...]}
        data = data.get("sections") or next((v for v in data.values() if isinstance(v, list)), [])
    return [{"title": str(s.get("title", "")), "text": str(s.get("text", ""))}
            for s in data if isinstance(s, dict)]
