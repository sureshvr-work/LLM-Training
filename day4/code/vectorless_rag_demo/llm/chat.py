"""
chat.py — send retrieved chunks to the LLM and get a cited answer.
Same prompt format as rag_demo1: each chunk is tagged [c#, p#] so the
model can cite the source inline.
"""
from openai import OpenAI

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


def _build_user_prompt(question: str, context_items: list[dict]) -> str:
    blocks = []
    for it in context_items:
        tag = f"[{it['id']}, p{it['page']}]"
        blocks.append(f"{tag} (source: {it['source']})\n{it['text']}")
    context = "\n\n".join(blocks)
    return f"Context passages:\n{context}\n\nQuestion: {question}"


def answer(question: str, context_items: list[dict],
           model: str = "gpt-4o-mini", sdk: str = "openai") -> str:
    user_prompt = _build_user_prompt(question, context_items)

    if sdk == "langchain":
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = ChatOpenAI(model=model, temperature=0)
        result = llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                             HumanMessage(content=user_prompt)])
        return result.content

    response = _openai().chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content
