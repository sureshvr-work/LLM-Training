"""
Drift & Hallucination Control Gateway
=======================================
Sits between the app and the LLM. Adds three layers on top of basic proxying:

  DRIFT DETECTION
    Every question is fingerprinted. On each call the response is compared
    to all previous responses for that question using cosine similarity on
    TF-IDF vectors (pure stdlib — no extra libraries).
    A drift_score close to 1.0 means stable; below DRIFT_THRESHOLD = flagged.

  HALLUCINATION GUARD — CONFIDENCE SCORING
    Injects a system prompt asking the model to append "Confidence: X/10"
    after its answer. The gateway parses that score. Anything below
    CONFIDENCE_THRESHOLD is flagged so the app can warn the user.

  HALLUCINATION GUARD — CONTEXT GROUNDING
    If the app sends a "context" field alongside the question, the gateway
    wraps the prompt with a RAG-style instruction: answer ONLY from this
    context, say "I don't know" if it isn't there.

  SELF-CONSISTENCY CHECK
    If check_consistency=true the gateway calls the LLM 3 times, computes
    pairwise similarity across all responses, and flags the result as
    unreliable when they disagree beyond the drift threshold.
"""

import os
import re
import json
import time
import hashlib
import math
from datetime import datetime, timezone
from collections import defaultdict, Counter

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Config — set via environment variables in docker-compose.yml
# ---------------------------------------------------------------------------
LLM_API_KEY          = os.environ.get("OPENAI_API_KEY", "")
LLM_URL              = "https://api.openai.com/v1/chat/completions"
REPORT_FILE          = "/app/reports/drift_report.jsonl"
DRIFT_THRESHOLD      = float(os.environ.get("DRIFT_THRESHOLD", "0.70"))
CONFIDENCE_THRESHOLD = int(os.environ.get("CONFIDENCE_THRESHOLD", "6"))
CONSISTENCY_CALLS    = int(os.environ.get("CONSISTENCY_CALLS", "3"))

app = FastAPI(title="Drift & Hallucination Gateway")

# Per-question response history: sha256(question)[:12] -> [response_text, ...]
response_history: dict[str, list[str]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Drift measurement — TF-IDF cosine similarity (no external deps)
# ---------------------------------------------------------------------------
# HOW IT WORKS:
#   1. Tokenise both strings into words.
#   2. Build term-frequency (TF) vectors for each.
#   3. Compute cosine similarity: dot(A,B) / (|A| * |B|).
#   Score 1.0 = identical, 0.0 = no shared words.
#   Repeated identical questions to an LLM typically score 0.7–0.95.
#   A drop below 0.70 usually means the answer has meaningfully changed.

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

def _tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {w: c / total for w, c in counts.items()}

def cosine_similarity(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    tfa, tfb = _tf(ta), _tf(tb)
    vocab = set(tfa) | set(tfb)
    dot   = sum(tfa.get(w, 0) * tfb.get(w, 0) for w in vocab)
    mag_a = math.sqrt(sum(v * v for v in tfa.values()))
    mag_b = math.sqrt(sum(v * v for v in tfb.values()))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

def avg_similarity_to_history(new_text: str, history: list[str]) -> float:
    """Average cosine similarity of new_text against all stored responses."""
    if not history:
        return 1.0   # first call: no baseline yet, score is "perfect"
    return sum(cosine_similarity(new_text, h) for h in history) / len(history)


# ---------------------------------------------------------------------------
# Confidence extraction
# ---------------------------------------------------------------------------
# The gateway injects a system prompt requesting "Confidence: X/10".
# This regex finds that line in the model's response.
_CONF_RE = re.compile(
    r"confidence[:\s]+(\d+)\s*/\s*10",
    re.IGNORECASE,
)

def extract_confidence(text: str) -> int | None:
    m = _CONF_RE.search(text)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------
async def call_llm(body: dict, extra_system: str | None = None) -> httpx.Response:
    """Forward a request to the real LLM, optionally prepending a system message."""
    if extra_system:
        body = dict(body)
        body["messages"] = [{"role": "system", "content": extra_system}] + list(body.get("messages", []))
    async with httpx.AsyncClient(timeout=40) as client:
        return await client.post(
            LLM_URL,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json=body,
        )

def extract_text(resp: httpx.Response) -> str:
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return ""

def extract_tokens(resp: httpx.Response) -> dict:
    """Pull prompt/completion/total token counts from the LLM response."""
    try:
        usage = resp.json().get("usage", {})
        return {
            "prompt_tokens":     usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens":      usage.get("total_tokens", 0),
        }
    except Exception:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ---------------------------------------------------------------------------
# Report helper — writes one JSON line per request
# ---------------------------------------------------------------------------
def report(record: dict):
    line = json.dumps(record)
    print("REPORT:", line, flush=True)
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    key_status = "SET" if LLM_API_KEY else "MISSING — calls will fail"
    print("=" * 60, flush=True)
    print("  Drift & Hallucination Gateway", flush=True)
    print("=" * 60, flush=True)
    print(f"  LLM URL              : {LLM_URL}", flush=True)
    print(f"  API key              : {key_status}", flush=True)
    print(f"  Drift threshold      : {DRIFT_THRESHOLD}", flush=True)
    print(f"  Confidence threshold : {CONFIDENCE_THRESHOLD}/10", flush=True)
    print(f"  Consistency calls    : {CONSISTENCY_CALLS}", flush=True)
    print(f"  Report file          : {REPORT_FILE}", flush=True)
    print("=" * 60, flush=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "drift-gateway running"}


@app.get("/drift-summary")
def drift_summary():
    """
    Return drift stats for every question seen so far.
    Call this after the app finishes to get a cross-question report.
    """
    result = {}
    for q_hash, history in response_history.items():
        if len(history) < 2:
            result[q_hash] = {"calls": len(history), "note": "need ≥2 calls to measure drift"}
            continue
        # Compare consecutive pairs to show how drift evolves call-by-call.
        pairwise = [
            round(cosine_similarity(history[i], history[i - 1]), 3)
            for i in range(1, len(history))
        ]
        result[q_hash] = {
            "calls": len(history),
            "pairwise_similarities": pairwise,
            "avg_similarity": round(sum(pairwise) / len(pairwise), 3),
            "min_similarity": round(min(pairwise), 3),
            "drift_detected": min(pairwise) < DRIFT_THRESHOLD,
        }
    return result


@app.post("/v1/chat/completions")
async def chat(request: Request):
    started  = time.time()
    body     = await request.json()

    # Pull caller-supplied extras (not passed to the LLM)
    context             = body.pop("context", None)          # grounding text
    check_consistency   = body.pop("check_consistency", False)
    check_hallucination = body.pop("check_hallucination", True)

    messages  = body.get("messages", [])
    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    q_hash    = hashlib.sha256(user_text.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Build the injected system prompt
    # ------------------------------------------------------------------
    system_parts = []

    # GROUNDING: constrain the model to the supplied context (RAG-style).
    # Without this, the model answers from training data and can hallucinate.
    if context:
        system_parts.append(
            "Answer ONLY using the context below. "
            "If the answer is not in the context, reply exactly: 'I don't know'.\n\n"
            f"CONTEXT:\n{context}"
        )

    # CONFIDENCE SCORING: ask the model to self-assess certainty.
    # This surfaces hallucination risk without a second API call.
    if check_hallucination:
        system_parts.append(
            "After your answer append a new line: Confidence: X/10  "
            "(10 = completely certain, 1 = mostly guessing)."
        )

    system_prompt = "\n\n".join(system_parts) if system_parts else None

    # ------------------------------------------------------------------
    # SELF-CONSISTENCY: call the LLM N times and compare agreement.
    # Inconsistent answers across identical prompts = hallucination risk.
    # ------------------------------------------------------------------
    if check_consistency:
        texts  = []
        tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for _ in range(CONSISTENCY_CALLS):
            r = await call_llm(body, system_prompt)
            texts.append(extract_text(r))
            t = extract_tokens(r)
            tokens["prompt_tokens"]     += t["prompt_tokens"]
            tokens["completion_tokens"] += t["completion_tokens"]
            tokens["total_tokens"]      += t["total_tokens"]

        pairs    = [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))]
        sims     = [cosine_similarity(texts[i], texts[j]) for i, j in pairs]
        avg_sim  = round(sum(sims) / len(sims), 3) if sims else 1.0
        is_consistent = avg_sim >= DRIFT_THRESHOLD

        report({
            "time": datetime.now(timezone.utc).isoformat(),
            "type": "consistency_check",
            "q_hash": q_hash,
            "calls": CONSISTENCY_CALLS,
            "pairwise_sims": [round(s, 3) for s in sims],
            "avg_similarity": avg_sim,
            "consistent": is_consistent,
            "tokens": tokens,
        })

        llm_text   = texts[0]   # use first response as the answer
        llm_status = 200
        warning    = None if is_consistent else {
            "type": "inconsistent_responses",
            "avg_similarity": avg_sim,
            "message": (
                f"Model gave inconsistent answers across {CONSISTENCY_CALLS} calls "
                f"(avg similarity {avg_sim} < threshold {DRIFT_THRESHOLD}). "
                "This response may be unreliable."
            ),
        }
    else:
        # ------------------------------------------------------------------
        # Single call path
        # ------------------------------------------------------------------
        r          = await call_llm(body, system_prompt)
        llm_text   = extract_text(r)
        llm_status = r.status_code
        tokens     = extract_tokens(r)
        warning    = None

    # ------------------------------------------------------------------
    # DRIFT DETECTION: compare this response to all previous ones.
    # ------------------------------------------------------------------
    history     = response_history[q_hash]
    drift_score = avg_similarity_to_history(llm_text, history)
    history.append(llm_text)

    drift_flagged = len(history) > 1 and drift_score < DRIFT_THRESHOLD

    # ------------------------------------------------------------------
    # CONFIDENCE PARSING
    # ------------------------------------------------------------------
    confidence        = extract_confidence(llm_text) if check_hallucination else None
    confidence_flagged = confidence is not None and confidence < CONFIDENCE_THRESHOLD

    elapsed = round(time.time() - started, 2)

    # Build the messages actually sent to the LLM (with injected system prompt).
    sent_messages = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    ) + body.get("messages", [])

    report({
        "time":               datetime.now(timezone.utc).isoformat(),
        "type":               "request",
        "q_hash":             q_hash,
        "call_number":        len(history),
        # ---- what was sent to the LLM ------------------------------------
        "llm_request": {
            "model":    body.get("model"),
            "messages": sent_messages,
        },
        # ---- what the LLM returned ---------------------------------------
        "llm_response": llm_text,
        # ---- token usage -------------------------------------------------
        "tokens":             tokens,
        # ---- guardrail results -------------------------------------------
        "drift_score":        round(drift_score, 3),
        "drift_flagged":      drift_flagged,
        "confidence":         confidence,
        "confidence_flagged": confidence_flagged,
        "grounded":           bool(context),
        "consistent":         not bool(warning),
        "seconds":            elapsed,
        "llm_status":         llm_status,
    })

    resp_content: dict = {
        "choices": [{"message": {"role": "assistant", "content": llm_text}}],
        "meta": {
            "drift_score":        round(drift_score, 3),
            "drift_flagged":      drift_flagged,
            "call_number":        len(history),
            "confidence":         confidence,
            "confidence_flagged": confidence_flagged,
            "grounded":           bool(context),
        },
    }
    if warning:
        resp_content["warning"] = warning

    return JSONResponse(status_code=llm_status, content=resp_content)
