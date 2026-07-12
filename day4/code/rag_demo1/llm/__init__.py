# The llm/ package is the only place that talks to a model provider.
# It exposes two pairs of functions; each can run via the OpenAI SDK
# directly or via LangChain, chosen by the `sdk` argument.
from .embed import embed_texts, embed_text
from .chat import answer

__all__ = ["embed_texts", "embed_text", "answer"]
