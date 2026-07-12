"""
embed.py  —  turn text into vectors, EITHER with the OpenAI SDK directly
OR through LangChain. Same inputs, same outputs; only the path differs.

This is the whole point of the SDK toggle in the UI: students see that
"OpenAI direct" and "LangChain" produce the same vectors — LangChain is a
wrapper, not magic.
"""
from openai import OpenAI

# Build the client lazily (on first use) so importing this module never
# requires the key — only an actual embed call does. Reads OPENAI_API_KEY.
_client = None


def _openai():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def embed_texts(texts, model="text-embedding-3-small", sdk="openai"):
    """Embed a LIST of strings -> list of vectors (each a list of floats)."""
    if sdk == "langchain":
        # ---- LangChain path ----
        from langchain_openai import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(model=model)
        return embedder.embed_documents(texts)

    # ---- OpenAI SDK direct path ----
    response = _openai().embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def embed_text(text, model="text-embedding-3-small", sdk="openai"):
    """Embed ONE string -> one vector. Used for the incoming query."""
    if sdk == "langchain":
        from langchain_openai import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(model=model)
        return embedder.embed_query(text)

    return embed_texts([text], model=model, sdk="openai")[0]
