"""
chat.py  —  ask the LLM to answer using ONLY the retrieved chunks,
EITHER with the OpenAI SDK directly OR through LangChain.

Same prompt, same result. The toggle shows the two abstraction levels:
raw client calls vs. LangChain's message objects.
"""
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
    # Each item: {id, page, source, text}. The tag carries the metadata the
    # model must cite; the source filename is shown so it can name the document.
    blocks = []
    for it in context_items:
        tag = f"[{it['id']}, p{it['page']}]"
        blocks.append(f"{tag} (source: {it['source']})\n{it['text']}")
    context = "\n\n".join(blocks)
    return f"Context passages:\n{context}\n\nQuestion: {question}"


def answer(question, context_items, model="gpt-4o-mini", sdk="openai"):
    """
    context_items : list of {id, page, source, text} — metadata travels into the prompt
    returns       : the model's answer (a string) with inline [c#, p#] citations
    """
    user_prompt = _build_user_prompt(question, context_items)

    if sdk == "langchain":
        # ---- LangChain path ----
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = ChatOpenAI(model=model, temperature=0)
        result = llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                             HumanMessage(content=user_prompt)])
        return result.content

    # ---- OpenAI SDK direct path ----
    response = _openai().chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content