"""
llm/extract.py — Demo 3: document -> structured JSON extraction.

System + user prompts come from the UI. Few-shot examples (if any) are sent as
REAL alternating turns (user: example doc -> assistant: example JSON) BEFORE the
real document turn — the production-faithful way to few-shot a chat model, not
pasted text. Per-provider message shapes still differ, which is the lesson:

  - turns:  OpenAI/Anthropic role user|assistant · Google role user|model · LangChain Human|AI
  - image:  OpenAI image_url · Anthropic source.base64 · Google inlineData
  - JSON:   OpenAI response_format · Google responseMimeType · Anthropic prompt-only
"""
import os
import re
import json
import httpx

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY", "")

MODEL_STR = {
    "openai":    "openai:gpt-4o-mini",
    "anthropic": "anthropic:claude-haiku-4-5",
    "google":    "google_genai:gemini-2.5-flash",
}
REAL_MODEL = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5", "google": "gemini-2.5-flash"}

DEFAULT_SYSTEM = (
    "You are a document data-extraction engine. Return ONLY one valid JSON object. "
    'For every requested field output {"value": <value or null>, "confidence": <0.0-1.0>}. '
    "Never invent data."
)

ECHO_DOC_LIMIT = 1500
ECHO_EX_LIMIT = 500


def _mask(k):
    return (k[:6] + "…" + k[-4:]) if len(k) > 12 else "…(not set)"


def _trim(s, n=ECHO_DOC_LIMIT):
    if not s:
        return s
    return s if len(s) <= n else s[:n] + f"\n…(+{len(s) - n} more chars — trimmed for display only)"


def _loads(s):
    if not s:
        return None, False
    t = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.MULTILINE).strip()
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b != -1:
        t = t[a:b + 1]
    try:
        return json.loads(t), True
    except Exception:
        return None, False


def _envelope(provider, interface, endpoint, auth, request_echo, response, raw, shots):
    parsed, ok = _loads(raw)
    return {
        "provider": provider.capitalize(),
        "model": MODEL_STR[provider] if interface == "langchain" else REAL_MODEL[provider],
        "interface": "LangChain" if interface == "langchain" else "Raw REST",
        "endpoint": endpoint, "auth": auth, "shots": shots,
        "request": request_echo, "response": response,
        "answer": raw, "extracted": parsed, "parse_ok": ok,
    }


def _real_user_openai(mode, user_prompt, text, images, omit_images):
    if mode == "image":
        blocks = [{"type": "text", "text": user_prompt}]
        for i, im in enumerate(images):
            url = (f"data:{im['media']};base64,<image {i+1} omitted>" if omit_images
                   else f"data:{im['media']};base64,{im['b64']}")
            blocks.append({"type": "image_url", "image_url": {"url": url}})
        return blocks
    body = f"{user_prompt}\n\nDOCUMENT:\n{_trim(text) if omit_images else text}"
    return body


# ---------------------------------------------------------------- RAW : OpenAI
async def _openai(mode, system, user_prompt, text, images, examples):
    endpoint = "https://api.openai.com/v1/chat/completions"
    msgs = [{"role": "system", "content": system}]
    emsgs = [{"role": "system", "content": system}]
    for ex in examples:
        a = {"role": "assistant", "content": ex["out"]}
        msgs += [{"role": "user", "content": "DOCUMENT:\n" + ex["doc"]}, a]
        emsgs += [{"role": "user", "content": "DOCUMENT:\n" + _trim(ex["doc"], ECHO_EX_LIMIT)}, a]
    msgs.append({"role": "user", "content": _real_user_openai(mode, user_prompt, text, images, False)})
    emsgs.append({"role": "user", "content": _real_user_openai(mode, user_prompt, text, images, True)})
    body = {"model": REAL_MODEL["openai"], "temperature": 0, "max_tokens": 1500,
            "response_format": {"type": "json_object"}, "messages": msgs}
    echo = {**body, "messages": emsgs}
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(endpoint, headers={"Authorization": f"Bearer {OPENAI_KEY}"}, json=body)
    r.raise_for_status()
    data = r.json()
    return _envelope("openai", "raw", f"POST {endpoint}",
                     f"Authorization: Bearer {_mask(OPENAI_KEY)} · response_format=json_object",
                     echo, data, data["choices"][0]["message"]["content"], len(examples))


# ------------------------------------------------------------- RAW : Anthropic
async def _anthropic(mode, system, user_prompt, text, images, examples):
    endpoint = "https://api.anthropic.com/v1/messages"
    if mode == "image":
        real = [{"type": "text", "text": user_prompt}] + [
            {"type": "image", "source": {"type": "base64", "media_type": im["media"], "data": im["b64"]}}
            for im in images]
        real_echo = [{"type": "text", "text": user_prompt}] + [
            {"type": "image", "source": {"type": "base64", "media_type": im["media"], "data": f"<image {i+1} omitted>"}}
            for i, im in enumerate(images)]
    else:
        real = f"{user_prompt}\n\nDOCUMENT:\n{text}"
        real_echo = f"{user_prompt}\n\nDOCUMENT:\n{_trim(text)}"
    msgs, emsgs = [], []
    for ex in examples:
        a = {"role": "assistant", "content": ex["out"]}
        msgs += [{"role": "user", "content": "DOCUMENT:\n" + ex["doc"]}, a]
        emsgs += [{"role": "user", "content": "DOCUMENT:\n" + _trim(ex["doc"], ECHO_EX_LIMIT)}, a]
    msgs.append({"role": "user", "content": real})
    emsgs.append({"role": "user", "content": real_echo})
    body = {"model": REAL_MODEL["anthropic"], "max_tokens": 1500, "temperature": 0,
            "system": system, "messages": msgs}
    echo = {**body, "messages": emsgs}
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(endpoint, headers={"x-api-key": ANTHROPIC_KEY,
                                            "anthropic-version": "2023-06-01",
                                            "content-type": "application/json"}, json=body)
    r.raise_for_status()
    data = r.json()
    return _envelope("anthropic", "raw", f"POST {endpoint}",
                     f"x-api-key: {_mask(ANTHROPIC_KEY)} · (no JSON flag — prompt only)",
                     echo, data, data["content"][0]["text"], len(examples))


# ---------------------------------------------------------------- RAW : Google
async def _google(mode, system, user_prompt, text, images, examples):
    model = REAL_MODEL["google"]
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    if mode == "image":
        real_parts = [{"text": user_prompt}] + [{"inlineData": {"mimeType": im["media"], "data": im["b64"]}} for im in images]
        echo_parts = [{"text": user_prompt}] + [
            {"inlineData": {"mimeType": im["media"], "data": f"<image {i+1} omitted>"}} for i, im in enumerate(images)]
    else:
        real_parts = [{"text": f"{user_prompt}\n\nDOCUMENT:\n{text}"}]
        echo_parts = [{"text": f"{user_prompt}\n\nDOCUMENT:\n{_trim(text)}"}]
    contents, econtents = [], []
    for ex in examples:
        m = {"role": "model", "parts": [{"text": ex["out"]}]}
        contents += [{"role": "user", "parts": [{"text": "DOCUMENT:\n" + ex["doc"]}]}, m]
        econtents += [{"role": "user", "parts": [{"text": "DOCUMENT:\n" + _trim(ex["doc"], ECHO_EX_LIMIT)}]}, m]
    contents.append({"role": "user", "parts": real_parts})
    econtents.append({"role": "user", "parts": echo_parts})
    body = {"contents": contents, "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"temperature": 0, "maxOutputTokens": 1500, "responseMimeType": "application/json"}}
    echo = {**body, "contents": econtents}
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(endpoint, headers={"x-goog-api-key": GOOGLE_KEY}, json=body)
    r.raise_for_status()
    data = r.json()
    raw = data["candidates"][0]["content"]["parts"][0]["text"]
    return _envelope("google", "raw", f"POST {endpoint}",
                     f"x-goog-api-key: {_mask(GOOGLE_KEY)} · responseMimeType=application/json",
                     echo, data, raw, len(examples))


# ------------------------------------------------------------------- LangChain
async def _langchain(provider, mode, system, user_prompt, text, images, examples):
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    llm = init_chat_model(MODEL_STR[provider], temperature=0, max_tokens=1500)
    real = _real_user_openai(mode, user_prompt, text, images, False)
    real_echo = _real_user_openai(mode, user_prompt, text, images, True)

    msgs = [SystemMessage(system)]
    emsgs = [{"role": "system", "content": system}]
    for ex in examples:
        msgs += [HumanMessage("DOCUMENT:\n" + ex["doc"]), AIMessage(ex["out"])]
        emsgs += [{"role": "human", "content": "DOCUMENT:\n" + _trim(ex["doc"], ECHO_EX_LIMIT)},
                  {"role": "ai", "content": ex["out"]}]
    msgs.append(HumanMessage(content=real))
    emsgs.append({"role": "human", "content": real_echo})

    resp = await llm.ainvoke(msgs)
    echo = {"init_chat_model": MODEL_STR[provider], "messages": emsgs}
    return _envelope(provider, "langchain",
                     f"init_chat_model('{MODEL_STR[provider]}') · await llm.ainvoke()",
                     "key read from env by the SDK",
                     echo,
                     {"content": resp.content, "usage_metadata": getattr(resp, "usage_metadata", None)},
                     resp.content, len(examples))


# -------------------------------------------------------------------- dispatch
async def run(interface, provider, mode, system, user_prompt, text=None, images=None, examples=None):
    if provider not in MODEL_STR:
        raise ValueError(f"unknown provider: {provider}")
    system = system or DEFAULT_SYSTEM
    images = images or []
    examples = examples or []
    if interface == "langchain":
        return await _langchain(provider, mode, system, user_prompt, text, images, examples)
    if provider == "openai":
        return await _openai(mode, system, user_prompt, text, images, examples)
    if provider == "anthropic":
        return await _anthropic(mode, system, user_prompt, text, images, examples)
    return await _google(mode, system, user_prompt, text, images, examples)
