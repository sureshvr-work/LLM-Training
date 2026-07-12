"""
Drift & Hallucination Test App
================================
Runs four experiments through the gateway. Each experiment isolates one
control technique so you can read the results side-by-side.

EXPERIMENT 1 — DRIFT DETECTION
  Asks the same question 5 times.
  The gateway computes a drift_score (cosine similarity vs. baseline).
  First call baseline = 1.0. Watch it fluctuate on later calls.
  Flag fires when score drops below 0.70.

EXPERIMENT 2 — CONFIDENCE SCORING (hallucination signal)
  The gateway injects a system prompt asking the model to rate its own
  certainty: "Confidence: X/10". Low scores flag likely hallucinations.
  We compare a real question vs. a made-up future event.

EXPERIMENT 3 — CONTEXT GROUNDING (RAG-style hallucination control)
  Same question, asked twice:
    a) Without context → model answers from training data, may hallucinate.
    b) With grounding context → model constrained to the supplied facts.
  Compare how the answers differ.

EXPERIMENT 4 — SELF-CONSISTENCY CHECK
  Gateway calls the LLM 3× with the same prompt, measures pairwise
  similarity. Disagreement = unreliable answer.
"""

import time
import httpx
import json

GATEWAY   = "http://gateway:8000"
CHAT_URL  = f"{GATEWAY}/v1/chat/completions"
HEALTH    = f"{GATEWAY}/health"
SUMMARY   = f"{GATEWAY}/drift-summary"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wait_for_gateway():
    for attempt in range(1, 25):
        try:
            httpx.get(HEALTH, timeout=2)
            print("Gateway ready.\n")
            return
        except Exception:
            print(f"Waiting for gateway... ({attempt})")
            time.sleep(1)
    raise SystemExit("Gateway never came up.")


def ask(
    question: str,
    context: str = None,
    check_consistency: bool = False,
    check_hallucination: bool = True,
    label: str = "",
) -> dict:
    """Send one question through the gateway, print a formatted result line."""
    payload = {
        "model":               "gpt-4o-mini",
        "messages":            [{"role": "user", "content": question}],
        "check_consistency":   check_consistency,
        "check_hallucination": check_hallucination,
    }
    if context:
        payload["context"] = context

    r    = httpx.post(CHAT_URL, json=payload, timeout=90)
    data = r.json()

    answer  = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    meta    = data.get("meta", {})
    warning = data.get("warning")

    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}Q: {question[:90]}")
    print(f"   Answer      : {answer[:150].strip()!r}")

    # Key metrics — this is what you read to measure drift & hallucination
    ds = meta.get("drift_score", "n/a")
    df = "⚠ DRIFT"     if meta.get("drift_flagged")      else "ok"
    cf = "⚠ LOW CONF"  if meta.get("confidence_flagged") else "ok"
    print(f"   drift_score : {ds}  [{df}]")
    print(f"   confidence  : {meta.get('confidence', 'n/a')}/10  [{cf}]")
    print(f"   call #      : {meta.get('call_number', 'n/a')}")
    print(f"   grounded    : {meta.get('grounded', False)}")
    if warning:
        print(f"   ⚠ WARNING   : {warning['message']}")

    return data


def section(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    wait_for_gateway()

    # ------------------------------------------------------------------ #
    # EXPERIMENT 1: DRIFT DETECTION
    #
    # HOW TO MEASURE DRIFT:
    #   drift_score = cosine_similarity(this_response, avg_of_all_prev)
    #   Range: 0.0 (completely different) → 1.0 (identical wording)
    #   Typical LLM range: 0.65–0.95 for the same question.
    #   Threshold (default 0.70): below this = drift flagged.
    #
    # WHAT TO WATCH:
    #   Call 1 → drift_score = 1.0 (no baseline yet)
    #   Call 2 → compares against call 1
    #   Call 3 → compares against avg of calls 1 & 2
    #   ...and so on. A dip signals the model is answering differently.
    # ------------------------------------------------------------------ #
    section("EXPERIMENT 1: DRIFT DETECTION")
    print("Asking the exact same question 5 times.")
    print("drift_score measures how similar each answer is to previous ones.")
    DRIFT_QUESTION = "Explain what machine learning is in exactly two sentences."
    for i in range(1, 6):
        ask(DRIFT_QUESTION, label=f"call {i}", check_hallucination=False)
        time.sleep(0.5)

    # ------------------------------------------------------------------ #
    # EXPERIMENT 2: CONFIDENCE SCORING
    #
    # HOW IT CONTROLS HALLUCINATION:
    #   The gateway injects: "After your answer, append: Confidence: X/10"
    #   The model's self-reported certainty correlates with hallucination.
    #   Low confidence (< 6/10) = flag and treat answer with skepticism.
    #
    # WHAT TO WATCH:
    #   Real question  → confidence 8–10, not flagged
    #   Made-up event  → confidence 1–4, flagged as unreliable
    # ------------------------------------------------------------------ #
    section("EXPERIMENT 2: CONFIDENCE SCORING (hallucination signal)")

    print("\n--- Real, well-known topic (expect HIGH confidence) ---")
    ask("What is the boiling point of water at sea level?", label="real topic")

    print("\n--- Made-up future event (expect LOW confidence) ---")
    ask(
        "Who won the 2031 Nobel Prize in Quantum Computing?",
        label="fake topic",
    )

    print("\n--- Ambiguous / contested claim (expect MEDIUM confidence) ---")
    ask(
        "What is the single best diet for human longevity?",
        label="contested topic",
    )

    # ------------------------------------------------------------------ #
    # EXPERIMENT 3: CONTEXT GROUNDING (RAG-style)
    #
    # HOW IT CONTROLS HALLUCINATION:
    #   Without context: model answers from training data → can hallucinate.
    #   With context:    gateway prepends "Answer ONLY from this context".
    #   The model must cite or stay within the provided facts.
    #   If the answer isn't in the context it should say "I don't know".
    #
    # WHAT TO WATCH:
    #   No-context answer: invented revenue figure (hallucination)
    #   With-context answer: exact figure from the context block
    # ------------------------------------------------------------------ #
    section("EXPERIMENT 3: CONTEXT GROUNDING (RAG-style)")

    COMPANY_FACTS = """
    Acme Corp Q3 2025 financial summary:
    - Total revenue   : $4.2 million
    - Net profit      : $310,000
    - Operating costs : $3.89 million
    - CEO             : Jane Smith
    - Headcount       : 42 full-time employees
    - Key product     : CloudSync Pro (launched July 2025)
    """

    print("\n--- Without context (model may hallucinate Acme's revenue) ---")
    ask("What was Acme Corp's total revenue in Q3 2025?", label="no context")

    print("\n--- With grounding context (model constrained to facts above) ---")
    ask(
        "What was Acme Corp's total revenue in Q3 2025?",
        context=COMPANY_FACTS,
        label="grounded",
    )

    print("\n--- Question NOT answerable from context (expect 'I don't know') ---")
    ask(
        "What was Acme Corp's revenue in Q4 2025?",
        context=COMPANY_FACTS,
        label="out-of-context",
    )

    # ------------------------------------------------------------------ #
    # EXPERIMENT 4: SELF-CONSISTENCY CHECK
    #
    # HOW IT CONTROLS HALLUCINATION:
    #   Gateway calls the LLM 3× with the same prompt.
    #   Computes pairwise cosine similarity across all 3 answers.
    #   If answers disagree (avg sim < threshold), flags as unreliable.
    #   Reliable factual answers should score > 0.80 across runs.
    #
    # WHAT TO WATCH:
    #   Subjective/opinion question → inconsistent, warning fires
    #   Factual question            → consistent, no warning
    # ------------------------------------------------------------------ #
    section("EXPERIMENT 4: SELF-CONSISTENCY CHECK")

    print("\n--- Factual question (should be consistent across 3 calls) ---")
    ask(
        "What is the chemical formula for table salt?",
        check_consistency=True,
        label="factual",
    )

    print("\n--- Subjective question (likely inconsistent across 3 calls) ---")
    ask(
        "What is the single best programming language to learn in 2025?",
        check_consistency=True,
        label="subjective",
    )

    # ------------------------------------------------------------------ #
    # DRIFT SUMMARY — full report across all questions
    # ------------------------------------------------------------------ #
    section("DRIFT SUMMARY (from gateway)")
    summary = httpx.get(SUMMARY, timeout=10).json()
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 65)
    print("Done. Reports saved to reports/drift_report.jsonl")
    print("=" * 65)
