"""All LLM/gateway work lives here so the controller stays thin.
One complete() handles every model — only the `model` string changes,
because LiteLLM does the provider translation. To add models you edit
litellm/config.yaml, not this file."""
import os
import httpx

LITELLM = os.environ.get("LITELLM_BASE_URL", "http://litellm:4000")
MASTER = os.environ.get("LITELLM_MASTER_KEY", "sk-1234")


async def issue_key(username: str) -> str:
    """Mint a per-student virtual key (master key never leaves the server)."""
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{LITELLM}/key/generate",
            headers={"Authorization": f"Bearer {MASTER}"},
            json={"max_budget": 1.0, "metadata": {"user": username}},
        )
    r.raise_for_status()
    return r.json()["key"]


async def complete(model, messages, temperature, max_tokens, top_p, key=None):
    """Forward an OpenAI-format chat request to LiteLLM. Works for any model."""
    payload = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(
            f"{LITELLM}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key or MASTER}"},
            json=payload,
        )
    r.raise_for_status()
    return r.json()
