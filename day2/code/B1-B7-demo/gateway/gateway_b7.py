"""
The Guardrail Gateway — B7 edition
====================================
Evolved from B1. Adds two response-side controls:

  RESPONSE CHECK 1 — PII scan (Presidio)
    Scans the LLM's reply for names, SSNs, card numbers, emails, etc.
    If found → 451 Unavailable For Legal Reasons. App never sees the PII.

  RESPONSE CHECK 2 — Schema validation (Pydantic)
    Ensures the LLM returned the shape the app expects.
    If malformed → 422 Unprocessable Entity.

Request-side controls from B1 are still here:
  RULE 1 — banned keyword → 403
  RULE 2 — prompt too large → 413
  TRACE  — every decision logged to trace.jsonl

The diff from B1 is small and deliberate — students can run:
  diff b1-gateway/gateway/gateway.py b7-gateway/gateway/gateway.py
"""
import os
import json
import time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from presidio_analyzer import AnalyzerEngine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLM_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
LLM_URL      = "https://api.openai.com/v1/chat/completions"
TRACE_FILE   = "/app/traces/trace.jsonl"

# ── Request-side rules (from B1) ────────────────────────────────────────────
BANNED_KEYWORDS  = ["projectfalcon", "project falcon", "acme-secret", "internal-only"]
MAX_PROMPT_CHARS = 8_000

# ── Response-side config (new in B7) ────────────────────────────────────────
# PII entity types to block. Full list at:
# https://microsoft.github.io/presidio/supported_entities/
BLOCKED_PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "US_BANK_NUMBER",
    "IBAN_CODE",
    "DATE_TIME",          # catches DOBs
    "US_PASSPORT",
    "MEDICAL_LICENSE",
]
PII_CONFIDENCE_THRESHOLD = 0.7   # ignore low-confidence hits

app = FastAPI(title="Guardrail Gateway — B7")

# Initialise Presidio once at startup — loading spaCy model takes ~2s
analyzer = AnalyzerEngine()


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    key_status = "SET" if LLM_API_KEY else "MISSING — calls will fail"
    print("=" * 60, flush=True)
    print("  Guardrail Gateway — B7 (request + response controls)", flush=True)
    print("=" * 60, flush=True)
    print(f"  LLM URL          : {LLM_URL}", flush=True)
    print(f"  API key          : {key_status}", flush=True)
    print(f"  Banned words     : {BANNED_KEYWORDS}", flush=True)
    print(f"  Max chars        : {MAX_PROMPT_CHARS:,}", flush=True)
    print(f"  PII entities     : {BLOCKED_PII_ENTITIES}", flush=True)
    print(f"  PII confidence   : {PII_CONFIDENCE_THRESHOLD}", flush=True)
    print(f"  Trace file       : {TRACE_FILE}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)


# ---------------------------------------------------------------------------
# Pydantic schema — what a valid LLM response must look like
# ---------------------------------------------------------------------------
class LLMMessage(BaseModel):
    role:    str
    content: str

class LLMChoice(BaseModel):
    index:   int
    message: LLMMessage

class LLMUsage(BaseModel):
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int

class LLMResponse(BaseModel):
    """
    The shape we expect from OpenAI for every chat completion.
    If the response does not match this, the gateway returns 422
    before the app sees anything.
    """
    id:      str
    object:  str
    model:   str
    choices: list[LLMChoice]
    usage:   LLMUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def trace(record: dict):
    """Print and append one trace line to the JSONL file."""
    line = json.dumps(record)
    print("TRACE:", line, flush=True)
    os.makedirs(os.path.dirname(TRACE_FILE), exist_ok=True)
    with open(TRACE_FILE, "a") as f:
        f.write(line + "\n")


def scan_pii(text: str) -> list[dict]:
    """
    Run Presidio over text. Return a list of findings above the
    confidence threshold. Each finding: {entity, text_snippet, score}.
    """
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=BLOCKED_PII_ENTITIES,
    )
    findings = []
    for r in results:
        if r.score >= PII_CONFIDENCE_THRESHOLD:
            snippet = text[r.start:r.end]
            findings.append({
                "entity": r.entity_type,
                "snippet": snippet,          # logged — never sent to app
                "score": round(r.score, 2),
            })
    return findings


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "gateway is running", "version": "B7"}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    started  = time.time()
    body     = await request.json()
    messages = body.get("messages", [])
    all_text = " ".join(m.get("content", "") for m in messages)
    lowered  = all_text.lower()

    # ── REQUEST CHECK 1: banned keyword ─────────────────────────────────────
    for word in BANNED_KEYWORDS:
        if word in lowered:
            trace({
                "time":     datetime.now(timezone.utc).isoformat(),
                "phase":    "request",
                "decision": "BLOCKED",
                "rule":     "banned-keyword",
                "detail":   word,
                "chars":    len(all_text),
            })
            return JSONResponse(
                status_code=403,
                content={"error": {
                    "type":    "guardrail_blocked",
                    "rule":    "banned-keyword",
                    "message": f"Request blocked: contains banned word '{word}'.",
                }},
            )

    # ── REQUEST CHECK 2: prompt too large ───────────────────────────────────
    if len(all_text) > MAX_PROMPT_CHARS:
        trace({
            "time":     datetime.now(timezone.utc).isoformat(),
            "phase":    "request",
            "decision": "BLOCKED",
            "rule":     "prompt-too-large",
            "chars":    len(all_text),
        })
        return JSONResponse(
            status_code=413,
            content={"error": {
                "type":    "guardrail_blocked",
                "rule":    "prompt-too-large",
                "message": f"Request blocked: {len(all_text):,} chars "
                           f"exceeds the {MAX_PROMPT_CHARS:,} char limit.",
            }},
        )

    # ── FORWARD to LLM ──────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            llm_response = await client.post(
                LLM_URL,
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                json=body,
            )
    except Exception as error:
        trace({
            "time":     datetime.now(timezone.utc).isoformat(),
            "phase":    "forward",
            "decision": "ERROR",
            "detail":   str(error),
        })
        return JSONResponse(
            status_code=502,
            content={"error": {
                "type":    "gateway_error",
                "message": f"Gateway could not reach the LLM: {error}",
            }},
        )

    # ── RESPONSE CHECK 1: schema validation (Pydantic) ──────────────────────
    try:
        raw = llm_response.json()
    except Exception:
        return JSONResponse(
            status_code=502,
            content={"error": {
                "type":    "llm_non_json_response",
                "message": llm_response.text[:300],
            }},
        )

    try:
        validated = LLMResponse(**raw)
    except ValidationError as ve:
        trace({
            "time":     datetime.now(timezone.utc).isoformat(),
            "phase":    "response",
            "decision": "BLOCKED",
            "rule":     "schema-validation",
            "detail":   ve.errors(),
        })
        return JSONResponse(
            status_code=422,
            content={"error": {
                "type":    "guardrail_blocked",
                "rule":    "schema-validation",
                "message": "LLM response did not match the expected schema.",
                "detail":  ve.errors(),
            }},
        )

    # ── RESPONSE CHECK 2: PII scan (Presidio) ───────────────────────────────
    reply_text = validated.choices[0].message.content
    pii_found  = scan_pii(reply_text)

    if pii_found:
        elapsed = round(time.time() - started, 2)
        trace({
            "time":     datetime.now(timezone.utc).isoformat(),
            "phase":    "response",
            "decision": "BLOCKED",
            "rule":     "pii-detected",
            "findings": pii_found,         # entity types + snippets logged
            "seconds":  elapsed,
        })
        return JSONResponse(
            status_code=451,               # 451 = Unavailable For Legal Reasons
            content={"error": {
                "type":    "guardrail_blocked",
                "rule":    "pii-detected",
                "message": "Response blocked: contains personally identifiable information.",
                "entities": [f["entity"] for f in pii_found],
                # NOTE: we do NOT return the snippets to the app —
                # that would defeat the purpose of blocking them.
            }},
        )

    # ── All checks passed — return to app ───────────────────────────────────
    elapsed = round(time.time() - started, 2)
    trace({
        "time":       datetime.now(timezone.utc).isoformat(),
        "phase":      "response",
        "decision":   "FORWARDED",
        "chars":      len(all_text),
        "reply_chars": len(reply_text),
        "llm_status": llm_response.status_code,
        "seconds":    elapsed,
    })
    return JSONResponse(
        status_code=llm_response.status_code,
        content=raw,
    )