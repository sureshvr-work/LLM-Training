"""
llm/langchain_iface.py — the SECOND interface.

Same three providers as providers.py, but called through LangChain instead
of raw REST. The contrast IS the lesson:

  providers.py (raw)      branches on EVERYTHING — endpoint, headers, request
                          shape, and where the answer lives in the response.

  this file (LangChain)   branches on ONE string ("provider:model"). The call
                          itself — .ainvoke(messages) — is identical for every
                          provider, and the response comes back normalized
                          (one .content field, one usage_metadata shape).

We return the SAME envelope shape as providers.py so the backend router and
the UI panels work unchanged:
  { provider, model, interface, endpoint, auth, request, response, answer }
"""
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# The ONLY thing that differs per provider: the "provider:model" string.
# (Note Google's prefix is "google_genai:" — it maps to langchain-google-genai.)
MODEL_STR = {
    "openai":    "openai:gpt-4o-mini",
    "anthropic": "anthropic:claude-haiku-4-5",
    "google":    "google_genai:gemini-2.5-flash",
}

# Which env var each provider's key is read from (LangChain reads these itself).
KEY_ENV = {
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google":    "GOOGLE_API_KEY",
}


async def run(provider, system, user, temperature, max_tokens, top_p):
    if provider not in MODEL_STR:
        raise ValueError(f"unknown provider: {provider}")

    # Only pass params the user actually set, so providers use their defaults.
    params = {}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if top_p is not None:
        params["top_p"] = top_p

    # 1) Build the model. This single line is the whole provider abstraction.
    llm = init_chat_model(MODEL_STR[provider], **params)

    # 2) Build messages the LangChain way — same classes for every provider.
    messages = []
    if system:
        messages.append(SystemMessage(system))
    messages.append(HumanMessage(user))

    # 3) Call it. IDENTICAL for OpenAI, Anthropic, Google.
    resp = await llm.ainvoke(messages)

    # What we "sent" — at the LangChain level, not the wire level.
    request = {
        "init_chat_model": MODEL_STR[provider],
        "params": params,
        "messages": [{"role": m.type, "content": m.content} for m in messages],
    }

    # What came back — already normalized by LangChain.
    response = {
        "content": resp.content,
        "usage_metadata": getattr(resp, "usage_metadata", None),
        "response_metadata": getattr(resp, "response_metadata", None),
    }

    return {
        "provider": provider.capitalize(),
        "model": MODEL_STR[provider],
        "interface": "LangChain",
        "endpoint": f"init_chat_model('{MODEL_STR[provider]}') · await llm.ainvoke(messages)",
        "auth": f"key read from env by the SDK · {KEY_ENV[provider]}",
        "request": request,
        "response": response,
        "answer": resp.content,
    }
