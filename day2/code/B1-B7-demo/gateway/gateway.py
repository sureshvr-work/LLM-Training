"""
The Guardrail Gateway
=====================
A tiny gateway that sits between a company app and an external LLM.

What it does, for every request:
  1. RULE 1 - block requests that contain a banned keyword
  2. RULE 2 - block requests that are too large (a simple token cap)
  3. TRACE  - log the request (size, decision, latency) to console + a file
  4. FORWARD - if the rules pass, send the request on to the real LLM

The real API key lives HERE, in the gateway - the app never sees it.

This is the explicit-gateway pattern: the app is configured to call this
gateway instead of the LLM directly. Same lesson as a TLS-intercepting
proxy, without the certificate setup.
"""
import os
import json
import time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Config - read from environment (set in docker-compose.yml)
# ---------------------------------------------------------------------------
LLM_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
LLM_URL      = "https://api.openai.com/v1/chat/completions"
TRACE_FILE   = "/app/traces/trace.jsonl"

# ---- our simple guardrail rules -------------------------------------------
BANNED_KEYWORDS = ["projectfalcon", "acme-secret", "internal-only"]
MAX_PROMPT_CHARS = 8000          # a simple stand-in for a token cap

app = FastAPI(title="Guardrail Gateway")

# ---------------------------------------------------------------------------
# Startup — print config so students see the gateway is alive and configured
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    key_status = "SET" if LLM_API_KEY else "MISSING — calls will fail"
    print("=" * 55, flush=True)
    print("  Guardrail gateway started", flush=True)
    print("=" * 55, flush=True)
    print(f"  LLM URL      : {LLM_URL}", flush=True)
    print(f"  API key      : {key_status}", flush=True)
    print(f"  Banned words : {BANNED_KEYWORDS}", flush=True)
    print(f"  Max chars    : {MAX_PROMPT_CHARS:,}", flush=True)
    print(f"  Trace file   : {TRACE_FILE}", flush=True)
    print("=" * 55, flush=True)
    print(flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def trace(record):
    """Print a trace line and append it to the trace file."""
    line = json.dumps(record)
    print("TRACE:", line, flush=True)
    os.makedirs(os.path.dirname(TRACE_FILE), exist_ok=True)
    with open(TRACE_FILE, "a") as f:
        f.write(line + "\n")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """A simple check that the gateway is up."""
    return {"status": "gateway is running"}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    """The one endpoint - the app sends its LLM request here."""
    started = time.time()
    body = await request.json()

    # Pull all the user text out of the request, lowercased, for checking.
    messages = body.get("messages", [])
    all_text = " ".join(m.get("content", "") for m in messages)
    lowered  = all_text.lower()

    # ---- RULE 1: banned keyword -------------------------------------------
    for word in BANNED_KEYWORDS:
        if word in lowered:
            trace({
                "time": datetime.now(timezone.utc).isoformat(),
                "decision": "BLOCKED",
                "rule": "banned-keyword",
                "detail": word,
                "chars": len(all_text),
            })
            return JSONResponse(
                status_code=403,
                content={"error": {
                    "type": "guardrail_blocked",
                    "rule": "banned-keyword",
                    "message": f"Request blocked: contains banned word '{word}'.",
                }},
            )

    # ---- RULE 2: prompt too large -----------------------------------------
    if len(all_text) > MAX_PROMPT_CHARS:
        trace({
            "time": datetime.now(timezone.utc).isoformat(),
            "decision": "BLOCKED",
            "rule": "prompt-too-large",
            "chars": len(all_text),
        })
        return JSONResponse(
            status_code=413,
            content={"error": {
                "type": "guardrail_blocked",
                "rule": "prompt-too-large",
                "message": f"Request blocked: {len(all_text)} chars over the "
                           f"{MAX_PROMPT_CHARS} limit.",
            }},
        )

    # ---- rules passed: FORWARD to the real LLM ----------------------------
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            llm_response = await client.post(
                LLM_URL,
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                json=body,
            )
    except Exception as error:
        # Could not reach the LLM at all (network, timeout, bad URL).
        trace({
            "time": datetime.now(timezone.utc).isoformat(),
            "decision": "ERROR",
            "detail": str(error),
        })
        return JSONResponse(
            status_code=502,
            content={"error": {
                "type": "gateway_error",
                "message": f"Gateway could not reach the LLM: {error}",
            }},
        )

    elapsed = round(time.time() - started, 2)
    trace({
        "time": datetime.now(timezone.utc).isoformat(),
        "decision": "FORWARDED",
        "chars": len(all_text),
        "llm_status": llm_response.status_code,
        "seconds": elapsed,
    })

    # Pass the LLM's response back to the app ( NOTE: THERE IS NO RESPONSE SCHEMA VALIDATION).
    # The LLM usually returns JSON,
    # but not always (e.g. some error pages) - so fall back to plain text.
    try:
        content = llm_response.json()
    except Exception:
        content = {"error": {
            "type": "llm_non_json_response",
            "message": llm_response.text[:300],
        }}
    return JSONResponse(status_code=llm_response.status_code, content=content)

