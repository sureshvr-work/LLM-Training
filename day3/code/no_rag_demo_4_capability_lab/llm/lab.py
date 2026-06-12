"""
llm/lab.py — Provider Capability Lab.

For each (capability, provider) it builds the EXACT request, renders a curl
command, executes it live, and returns the model output. The request/curl is
the teaching artifact — it is constructed faithfully even if the live call is
not run. LangChain cells return the unified-shape Python instead of curl.

Capabilities: text · pdf · image · audio · files  (files = upload then reuse by id)
Providers:    openai · anthropic · google · langchain
"""
import os
import re
import json
import base64
import httpx

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY", "")

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TRANSCRIBE = "gpt-4o-mini-transcribe"
ANTHROPIC_MODEL = "claude-haiku-4-5"
GOOGLE_MODEL = "gemini-2.5-flash"
LC_MODEL = {"openai": "openai:gpt-4o-mini", "anthropic": "anthropic:claude-haiku-4-5", "google": "google_genai:gemini-2.5-flash"}

OPENAI_RESP = "https://api.openai.com/v1/responses"
OPENAI_FILES = "https://api.openai.com/v1/files"
OPENAI_AUDIO = "https://api.openai.com/v1/audio/transcriptions"
ANTHROPIC_MSG = "https://api.anthropic.com/v1/messages"
ANTHROPIC_FILES = "https://api.anthropic.com/v1/files"
GOOGLE_GEN = f"https://generativelanguage.googleapis.com/v1beta/models/{GOOGLE_MODEL}:generateContent"

DEFAULT_PROMPT = {
    "text":  "Summarize this document in one sentence.",
    "pdf":   "What is this document about? One sentence.",
    "image": "Describe this image in one sentence.",
    "audio": "Transcribe this audio.",
    "files": "What is this document about? One sentence.",
}


# ----------------------------------------------------------------- helpers
def _mask_key():
    return "$OPENAI_API_KEY"


def _trunc(obj):
    """Recursively shorten long base64-ish strings for display."""
    if isinstance(obj, dict):
        return {k: _trunc(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_trunc(v) for v in obj]
    if isinstance(obj, str) and len(obj) > 90:
        return obj[:40] + "…<base64 truncated>…"
    return obj


def _curl(method, url, headers, body=None, form=None, files=None):
    lines = [f"curl -X {method} {url}"]
    for k, v in headers.items():
        lines.append(f"  -H '{k}: {v}'")
    if body is not None:
        lines.append("  -d '" + json.dumps(_trunc(body)) + "'")
    if form:
        for k, v in form.items():
            lines.append(f"  -F '{k}={v}'")
    if files:
        for k, v in files.items():
            lines.append(f"  -F '{k}=@{v}'")
    return " \\\n".join(lines)


def _env(provider, capability, *, model="", supported=True, steps=None,
         output="", error="", note="", lc_code="", endpoint=""):
    return {
        "provider": provider, "capability": capability, "model": model,
        "supported": supported, "endpoint": endpoint,
        "steps": steps or [], "output": output, "error": error,
        "note": note, "lc_code": lc_code,
    }


async def _post_json(url, headers, body):
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(url, headers=headers, json=body)
    r.raise_for_status()
    return r.json()


# =================================================================== OpenAI
async def _openai(cap, prompt, b64, media, text):
    if cap == "audio":
        # transcription endpoint (multipart)
        disp_headers = {"Authorization": f"Bearer {_mask_key()}"}
        curl = _curl("POST", OPENAI_AUDIO, disp_headers,
                     form={"model": OPENAI_TRANSCRIBE}, files={"file": "audio.mp3"})
        step = {"label": "Transcribe (multipart upload)", "endpoint": f"POST {OPENAI_AUDIO}",
                "curl": curl, "request": {"model": OPENAI_TRANSCRIBE, "file": "<audio bytes>"}}
        try:
            raw = base64.b64decode(b64)
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(OPENAI_AUDIO, headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                                 data={"model": OPENAI_TRANSCRIBE},
                                 files={"file": ("audio." + (media.split("/")[-1] or "mp3"), raw, media or "audio/mpeg")})
            r.raise_for_status()
            out = r.json().get("text", "")
        except Exception as e:
            return _env("openai", cap, model=OPENAI_TRANSCRIBE, steps=[step], error=str(e),
                        endpoint=f"POST {OPENAI_AUDIO}",
                        note="Audio goes to the transcription endpoint, not the chat model. gpt-4o-audio can also reason over audio inline.")
        return _env("openai", cap, model=OPENAI_TRANSCRIBE, steps=[step], output=out,
                    endpoint=f"POST {OPENAI_AUDIO}",
                    note="Audio goes to the transcription endpoint, not the chat model.")

    # text / pdf / image  -> Responses API
    if cap == "text":
        content = [{"type": "input_text", "text": f"{prompt}\n\nDOCUMENT:\n{text}"}]
    elif cap == "pdf":
        content = [{"type": "input_text", "text": prompt},
                   {"type": "input_file", "filename": "document.pdf", "file_data": f"data:application/pdf;base64,{b64}"}]
    else:  # image
        content = [{"type": "input_text", "text": prompt},
                   {"type": "input_image", "image_url": f"data:{media};base64,{b64}"}]
    body = {"model": OPENAI_MODEL, "input": [{"role": "user", "content": content}]}
    disp_headers = {"Authorization": f"Bearer {_mask_key()}", "Content-Type": "application/json"}
    step = {"label": "Generate (Responses API)", "endpoint": f"POST {OPENAI_RESP}",
            "curl": _curl("POST", OPENAI_RESP, disp_headers, body=body), "request": _trunc(body)}
    try:
        data = await _post_json(OPENAI_RESP, {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}, body)
        out = data.get("output_text") or _openai_extract(data)
    except Exception as e:
        return _env("openai", cap, model=OPENAI_MODEL, steps=[step], error=str(e), endpoint=f"POST {OPENAI_RESP}")
    return _env("openai", cap, model=OPENAI_MODEL, steps=[step], output=out, endpoint=f"POST {OPENAI_RESP}")


def _openai_extract(data):
    try:
        for item in data.get("output", []):
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    return c.get("text", "")
    except Exception:
        pass
    return json.dumps(data)[:400]


async def _openai_files(cap, prompt, b64, media):
    # step 1: upload
    raw = base64.b64decode(b64)
    up_disp = {"Authorization": f"Bearer {_mask_key()}"}
    up_curl = _curl("POST", OPENAI_FILES, up_disp, form={"purpose": "user_data"}, files={"file": "document.pdf"})
    step1 = {"label": "1 · Upload file", "endpoint": f"POST {OPENAI_FILES}", "curl": up_curl,
             "request": {"purpose": "user_data", "file": "<pdf bytes>"}}
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(OPENAI_FILES, headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                             data={"purpose": "user_data"},
                             files={"file": ("document.pdf", raw, "application/pdf")})
        r.raise_for_status()
        file_id = r.json()["id"]
    except Exception as e:
        return _env("openai", cap, model=OPENAI_MODEL, steps=[step1], error=str(e), endpoint=f"POST {OPENAI_FILES}",
                    note="Upload once, then reference by file_id across many calls.")
    # step 2: use file_id
    body = {"model": OPENAI_MODEL, "input": [{"role": "user", "content": [
        {"type": "input_text", "text": prompt},
        {"type": "input_file", "file_id": file_id}]}]}
    disp = {"Authorization": f"Bearer {_mask_key()}", "Content-Type": "application/json"}
    step2 = {"label": "2 · Reference by file_id", "endpoint": f"POST {OPENAI_RESP}",
             "curl": _curl("POST", OPENAI_RESP, disp, body=body), "request": body}
    try:
        data = await _post_json(OPENAI_RESP, {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}, body)
        out = data.get("output_text") or _openai_extract(data)
    except Exception as e:
        return _env("openai", cap, model=OPENAI_MODEL, steps=[step1, step2], error=str(e), endpoint=f"POST {OPENAI_RESP}")
    return _env("openai", cap, model=OPENAI_MODEL, steps=[step1, step2], output=out, endpoint=f"POST {OPENAI_RESP}",
                note="Upload once, reuse the file_id across many calls — no re-encoding.")


# ================================================================ Anthropic
async def _anthropic(cap, prompt, b64, media, text):
    if cap == "audio":
        code = ('# Claude has no native audio input. Transcribe first, then send text:\n'
                'transcript = whisper(audio)            # e.g. OpenAI / Deepgram / AssemblyAI\n'
                'client.messages.create(\n'
                '    model="%s", max_tokens=512,\n'
                '    messages=[{"role": "user", "content": f"Transcript:\\n{transcript}"}])' % ANTHROPIC_MODEL)
        return _env("anthropic", cap, model=ANTHROPIC_MODEL, supported=False, lc_code=code,
                    note="Anthropic has no native audio/video input. The pattern is: transcribe with a speech model, then send the text to Claude.")

    if cap == "text":
        content = f"{prompt}\n\nDOCUMENT:\n{text}"
    elif cap == "pdf":
        content = [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}},
                   {"type": "text", "text": prompt}]
    else:  # image
        content = [{"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                   {"type": "text", "text": prompt}]
    body = {"model": ANTHROPIC_MODEL, "max_tokens": 512, "messages": [{"role": "user", "content": content}]}
    disp = {"x-api-key": "$ANTHROPIC_API_KEY", "anthropic-version": "2023-06-01", "content-type": "application/json"}
    step = {"label": "Generate (Messages API)", "endpoint": f"POST {ANTHROPIC_MSG}",
            "curl": _curl("POST", ANTHROPIC_MSG, disp, body=body), "request": _trunc(body)}
    try:
        data = await _post_json(ANTHROPIC_MSG, {"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}, body)
        out = data["content"][0]["text"]
    except Exception as e:
        return _env("anthropic", cap, model=ANTHROPIC_MODEL, steps=[step], error=str(e), endpoint=f"POST {ANTHROPIC_MSG}")
    return _env("anthropic", cap, model=ANTHROPIC_MODEL, steps=[step], output=out, endpoint=f"POST {ANTHROPIC_MSG}")


async def _anthropic_files(cap, prompt, b64, media):
    raw = base64.b64decode(b64)
    up_disp = {"x-api-key": "$ANTHROPIC_API_KEY", "anthropic-version": "2023-06-01", "anthropic-beta": "files-api-2025-04-14"}
    up_curl = _curl("POST", ANTHROPIC_FILES, up_disp, files={"file": "document.pdf"})
    step1 = {"label": "1 · Upload file (beta)", "endpoint": f"POST {ANTHROPIC_FILES}", "curl": up_curl,
             "request": {"file": "<pdf bytes>"}}
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(ANTHROPIC_FILES, headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "anthropic-beta": "files-api-2025-04-14"},
                             files={"file": ("document.pdf", raw, "application/pdf")})
        r.raise_for_status()
        file_id = r.json()["id"]
    except Exception as e:
        return _env("anthropic", cap, model=ANTHROPIC_MODEL, steps=[step1], error=str(e), endpoint=f"POST {ANTHROPIC_FILES}",
                    note="Files API is beta — needs the anthropic-beta: files-api-2025-04-14 header.")
    body = {"model": ANTHROPIC_MODEL, "max_tokens": 512, "messages": [{"role": "user", "content": [
        {"type": "document", "source": {"type": "file", "file_id": file_id}},
        {"type": "text", "text": prompt}]}]}
    disp = {"x-api-key": "$ANTHROPIC_API_KEY", "anthropic-version": "2023-06-01", "anthropic-beta": "files-api-2025-04-14", "content-type": "application/json"}
    step2 = {"label": "2 · Reference by file_id", "endpoint": f"POST {ANTHROPIC_MSG}",
             "curl": _curl("POST", ANTHROPIC_MSG, disp, body=body), "request": body}
    try:
        data = await _post_json(ANTHROPIC_MSG, {"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "anthropic-beta": "files-api-2025-04-14", "content-type": "application/json"}, body)
        out = data["content"][0]["text"]
    except Exception as e:
        return _env("anthropic", cap, model=ANTHROPIC_MODEL, steps=[step1, step2], error=str(e), endpoint=f"POST {ANTHROPIC_MSG}")
    return _env("anthropic", cap, model=ANTHROPIC_MODEL, steps=[step1, step2], output=out, endpoint=f"POST {ANTHROPIC_MSG}",
                note="Files API is beta — upload once, reference the file_id in a document block.")


# =================================================================== Google
async def _google(cap, prompt, b64, media, text):
    if cap == "text":
        parts = [{"text": f"{prompt}\n\nDOCUMENT:\n{text}"}]
    elif cap == "pdf":
        parts = [{"inlineData": {"mimeType": "application/pdf", "data": b64}}, {"text": prompt}]
    elif cap == "image":
        parts = [{"inlineData": {"mimeType": media, "data": b64}}, {"text": prompt}]
    else:  # audio — native
        parts = [{"inlineData": {"mimeType": media or "audio/mpeg", "data": b64}}, {"text": prompt}]
    body = {"contents": [{"role": "user", "parts": parts}]}
    disp = {"x-goog-api-key": "$GOOGLE_API_KEY", "Content-Type": "application/json"}
    step = {"label": "generateContent", "endpoint": f"POST {GOOGLE_GEN}",
            "curl": _curl("POST", GOOGLE_GEN, disp, body=body), "request": _trunc(body)}
    try:
        data = await _post_json(GOOGLE_GEN, {"x-goog-api-key": GOOGLE_KEY, "Content-Type": "application/json"}, body)
        out = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return _env("google", cap, model=GOOGLE_MODEL, steps=[step], error=str(e), endpoint=f"POST {GOOGLE_GEN}",
                    note=("Gemini reads audio/video natively via inlineData." if cap == "audio" else ""))
    return _env("google", cap, model=GOOGLE_MODEL, steps=[step], output=out, endpoint=f"POST {GOOGLE_GEN}",
                note=("Gemini reads audio/video natively via inlineData — no separate transcription step." if cap == "audio" else ""))


async def _google_files(cap, prompt, b64, media):
    code = ('from google import genai\n'
            'client = genai.Client()\n'
            '# Resumable upload via the SDK (raw REST is a 2-step resumable protocol)\n'
            'f = client.files.upload(file="document.pdf")     # -> file URI, expires ~48h\n'
            'client.models.generate_content(\n'
            '    model="%s",\n'
            '    contents=[f, "%s"])' % (GOOGLE_MODEL, DEFAULT_PROMPT["files"]))
    return _env("google", cap, model=GOOGLE_MODEL, supported=True, lc_code=code, endpoint="File API (SDK)",
                note="Gemini's File API upload is a resumable protocol — shown via the SDK. Returns a file URI that expires after ~48h.")


# ================================================================ LangChain
async def _langchain(cap, provider_hint, prompt, b64, media, text):
    target = provider_hint if provider_hint in LC_MODEL else "openai"
    model_str = LC_MODEL[target]
    if cap == "text":
        code = ('from langchain.chat_models import init_chat_model\n'
                'from langchain_core.messages import HumanMessage\n'
                'llm = init_chat_model("%s")\n'
                'llm.invoke([HumanMessage("%s\\n\\n" + text)])' % (model_str, prompt))
    elif cap in ("pdf", "image", "files"):
        block = "image" if cap == "image" else "file"
        code = ('from langchain.chat_models import init_chat_model\n'
                'from langchain_core.messages import HumanMessage\n'
                'llm = init_chat_model("%s")\n'
                'msg = HumanMessage(content=[\n'
                '    {"type": "text", "text": "%s"},\n'
                '    {"type": "%s", "source_type": "base64",\n'
                '     "data": b64, "mime_type": "%s"},\n'
                '])\n'
                'llm.invoke([msg])   # same code, any provider' % (model_str, prompt, block, media or "application/pdf"))
    else:  # audio
        code = ('from langchain_community.document_loaders.parsers import OpenAIWhisperParser\n'
                '# LangChain has no chat audio block — use an audio loader/parser to get text:\n'
                'transcript = OpenAIWhisperParser().lazy_parse(blob)\n'
                'llm.invoke([HumanMessage("Transcript:\\n" + transcript)])')

    env = _env("langchain", cap, model=model_str, lc_code=code, endpoint=f"init_chat_model('{model_str}')",
               note="One message shape; LangChain translates it to each provider's wire format underneath.")
    # best-effort live run for text/image/pdf
    if cap in ("text", "image", "pdf"):
        try:
            from langchain.chat_models import init_chat_model
            from langchain_core.messages import HumanMessage
            llm = init_chat_model(model_str, max_tokens=512)
            if cap == "text":
                msg = HumanMessage(f"{prompt}\n\nDOCUMENT:\n{text}")
            else:
                blk = "image" if cap == "image" else "file"
                msg = HumanMessage(content=[{"type": "text", "text": prompt},
                                            {"type": blk, "source_type": "base64", "data": b64,
                                             "mime_type": media or "application/pdf"}])
            resp = await llm.ainvoke([msg])
            env["output"] = resp.content if isinstance(resp.content, str) else json.dumps(resp.content)[:400]
        except Exception as e:
            env["error"] = str(e)
    return env


# =================================================================== dispatch
async def run(capability, provider, kind=None, b64=None, media=None, text=None, prompt=None, lc_target="openai"):
    prompt = prompt or DEFAULT_PROMPT.get(capability, "Describe the input.")
    if provider == "langchain":
        return await _langchain(capability, lc_target, prompt, b64, media, text)
    if capability == "files":
        if provider == "openai":
            return await _openai_files(capability, prompt, b64, media)
        if provider == "anthropic":
            return await _anthropic_files(capability, prompt, b64, media)
        return await _google_files(capability, prompt, b64, media)
    if provider == "openai":
        return await _openai(capability, prompt, b64, media, text)
    if provider == "anthropic":
        return await _anthropic(capability, prompt, b64, media, text)
    if provider == "google":
        return await _google(capability, prompt, b64, media, text)
    raise ValueError(f"unknown provider {provider}")
