"""
llm/providers.py — the LLM layer. The whole lesson lives here.

Three providers. Three DIFFERENT request shapes going in, three DIFFERENT
response shapes coming out. We do NOT hide the differences behind a
normalization layer — seeing the differences IS the point.

This is library code: the backend imports it. It is NOT its own container,
because it is just functions the backend calls.

Each call_* is written in the same shape on purpose, so students can read
them side by side and spot exactly what changes:

                OpenAI                Anthropic              Google (Gemini)
  endpoint      /v1/chat/completions  /v1/messages           /models/<m>:generateContent
  auth header   Authorization: Bearer x-api-key               x-goog-api-key
  system msg    a "system" message    top-level "system"      "systemInstruction"
  user msg      messages[].content    messages[].content      contents[].parts[].text
  max tokens    optional max_tokens   REQUIRED max_tokens     maxOutputTokens
  answer is at  choices[0].message    content[0].text         candidates[0].content
                .content                                      .parts[0].text

Every call_* returns the SAME envelope so the UI can show what went in and
what came back for whichever model the student picked:
  { provider, model, endpoint, auth, request, response, answer }
"""
import os
import httpx

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY", "")


def _mask(key: str) -> str:
    """Never show a real key in the UI."""
    return (key[:6] + "…" + key[-4:]) if len(key) > 12 else "…(not set)"


# ----------------------------------------------------------------------------
# 1) OpenAI  —  POST /v1/chat/completions
# ----------------------------------------------------------------------------
async def call_openai(system, user, temperature, max_tokens, top_p):
    model = "gpt-4o-mini"
    endpoint = "https://api.openai.com/v1/chat/completions"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    request = {"model": model, "messages": messages}
    if temperature is not None:
        request["temperature"] = temperature
    if max_tokens is not None:
        request["max_tokens"] = max_tokens          # optional for OpenAI
    if top_p is not None:
        request["top_p"] = top_p

    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(
            endpoint,
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            json=request,
        )
    r.raise_for_status()
    response = r.json()
    answer = response["choices"][0]["message"]["content"]

    return {
        "provider": "OpenAI",
        "model": model,
        "endpoint": endpoint,
        "auth": f"Authorization: Bearer {_mask(OPENAI_KEY)}",
        "request": request,
        "response": response,
        "answer": answer,
    }


# ----------------------------------------------------------------------------
# 2) Anthropic  —  POST /v1/messages
# ----------------------------------------------------------------------------
async def call_anthropic(system, user, temperature, max_tokens, top_p):
    model = "claude-haiku-4-5"
    endpoint = "https://api.anthropic.com/v1/messages"

    # Note the differences vs OpenAI:
    #   - system prompt is a TOP-LEVEL field, not a message
    #   - max_tokens is REQUIRED
    request = {
        "model": model,
        "max_tokens": max_tokens or 1024,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        request["system"] = system
    if temperature is not None:
        request["temperature"] = temperature
    if top_p is not None:
        request["top_p"] = top_p

    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(
            endpoint,
            headers={
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=request,
        )
    r.raise_for_status()
    response = r.json()
    answer = response["content"][0]["text"]          # different path!

    return {
        "provider": "Anthropic",
        "model": model,
        "endpoint": endpoint,
        "auth": f"x-api-key: {_mask(ANTHROPIC_KEY)}  ·  anthropic-version: 2023-06-01",
        "request": request,
        "response": response,
        "answer": answer,
    }


# ----------------------------------------------------------------------------
# 3) Google Gemini  —  POST /models/<model>:generateContent  (NATIVE, not the
#    OpenAI-compatible shim — we want students to see the real difference)
# ----------------------------------------------------------------------------
async def call_google(system, user, temperature, max_tokens, top_p):
    model = "gemini-2.5-flash"
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )

    # Completely different shape: contents -> parts -> text
    request = {"contents": [{"role": "user", "parts": [{"text": user}]}]}
    if system:
        request["systemInstruction"] = {"parts": [{"text": system}]}
    gen = {}
    if temperature is not None:
        gen["temperature"] = temperature
    if max_tokens is not None:
        gen["maxOutputTokens"] = max_tokens
    if top_p is not None:
        gen["topP"] = top_p                          # Gemini calls it topP
    if gen:
        request["generationConfig"] = gen

    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(
            endpoint,
            headers={"x-goog-api-key": GOOGLE_KEY},
            json=request,
        )
    r.raise_for_status()
    response = r.json()
    answer = response["candidates"][0]["content"]["parts"][0]["text"]  # different again!

    return {
        "provider": "Google",
        "model": model,
        "endpoint": endpoint,
        "auth": f"x-goog-api-key: {_mask(GOOGLE_KEY)}",
        "request": request,
        "response": response,
        "answer": answer,
    }


# ----------------------------------------------------------------------------
# Dispatcher — picks the function based on what the student chose in the UI.
# ----------------------------------------------------------------------------
PROVIDERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
}


async def run(provider, system, user, temperature, max_tokens, top_p):
    fn = PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(f"unknown provider: {provider}")
    return await fn(system, user, temperature, max_tokens, top_p)
