# The llm/ package is the ONLY place that talks to a model provider.
# Keyword retrieval needs no model at all — this package is used only for the
# final answer (and, optionally, the LLM-driven chunker). Each function runs via
# the OpenAI SDK directly or via LangChain, chosen by the `sdk` argument, so the
# room sees that LangChain is a wrapper, not magic.
from .chat import answer, segment

__all__ = ["answer", "segment"]
