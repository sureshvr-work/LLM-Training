"""
The Company App — B7 edition
==============================
Tests all five gateway behaviours on one endpoint:

  200  normal             → clean request, clean response
  403  banned keyword     → blocked at request
  413  too large          → blocked at request
  400  BLOCK PII          → card/account numbers hard-stopped
  200  ANONYMIZE PII      → SSN/name replaced, LLM still answers
  451  PII in response    → LLM generated PII, blocked on way back
  200  clean              → passes both fences
"""

import time
import httpx

GATEWAY_URL    = "http://gateway:8000/v1/chat/completions"
GATEWAY_HEALTH = "http://gateway:8000/health"


def wait_for_gateway():
    """Poll /health until gateway is ready (uvicorn finishes booting)."""
    for attempt in range(1, 21):
        try:
            httpx.get(GATEWAY_HEALTH, timeout=2)
            print("Gateway is ready.\n")
            return
        except Exception:
            print(f"Waiting for gateway... ({attempt})")
            time.sleep(1)
    raise SystemExit("Gateway never came up.")


def ask(label: str, question: str, expected: str, note: str = ""):
    """Send one question through the gateway and print the result clearly."""
    print(f"{'─' * 65}")
    print(f"TEST : {label}")
    print(f"  Expected : HTTP {expected}")
    if note:
        print(f"  Note     : {note}")
    print(f"  Question : {question[:90]}{'...' if len(question) > 90 else ''}")

    response = httpx.post(
        GATEWAY_URL,
        json={
            "model":    "gpt-4o-mini",
            "messages": [{"role": "user", "content": question}],
        },
        timeout=40,
    )

    status = response.status_code
    data   = response.json()
    match  = "✓ PASS" if str(status) == expected else "✗ FAIL"

    print(f"  Got      : HTTP {status}  {match}")

    if status == 200:
        meta  = data.get("gateway_meta", {})
        reply = data["choices"][0]["message"]["content"]
        if meta.get("anonymized"):
            print(f"  Anonymized : YES")
            print(f"  Sent as    : {(meta.get('clean_text') or '')[:80]}...")
        else:
            print(f"  Anonymized : NO — clean request")
        print(f"  Reply      : {reply.strip()[:120]}")

    elif status == 400:
        err = data.get("error", {})
        print(f"  Blocked  : {err.get('message')}")
        print(f"  Entities : {err.get('entities')}")
        print(f"  Lesson   : PII never reached OpenAI — $0 cost, data stayed local")

    elif status == 451:
        err = data.get("error", {})
        print(f"  Blocked  : {err.get('message')}")
        print(f"  Entities : {err.get('entities')}")
        print(f"  Lesson   : LLM already ran and was paid for — but output was stopped")

    elif status == 422:
        err = data.get("error", {})
        print(f"  Blocked  : {err.get('message')}")

    elif status == 403:
        err = data.get("error", {})
        print(f"  Blocked  : {err.get('message')}")

    elif status == 413:
        err = data.get("error", {})
        print(f"  Blocked  : {err.get('message')}")

    else:
             print(f"  Response   : {data}")

    print()


if __name__ == "__main__":
    print("=" * 65)
    print("Company app — B7 (bidirectional PII fence (block + anonymize + response fence) + schema validation)")
    print("=" * 65)
    print()

    wait_for_gateway()

    # ── B1 controls: request-side (unchanged, still work) ───────────────────

    ask(
        label    = "Normal banking question",
        question = "In one sentence, what is a fixed deposit?",
        expected = "200",
    )

    ask(
        label    = "Banned keyword in request",
        question = "Tell me about projectfalcon, our new release.",
        expected = "403",
    )

    ask(
        label    = "Prompt too large",
        question = "Summarize this: " + ("data " * 3000),
        expected = "413",
    )

    # ── B7 new: REQUEST-side PII fence ──────────────────────────────────────

    ask(
        label    = "BLOCK — credit card number in request",
        question = (
            "My credit card 4111-1111-1111-1111 was declined. "
            "What should I do?"
        ),
        expected = "400",
        note     = "CREDIT_CARD is a hard-stop entity. Never reaches OpenAI.",
    )

    ask(
        label    = "BLOCK — app leaks raw customer record",
        question = (
            "Customer IBAN: GB29 NWBK 6016 1331 9268 19. "
            "Write a one-sentence summary for the case file."
        ),
        expected = "400",
        note     = "IBAN_CODE is a hard-stop entity. Detected reliably by Presidio.",
    )

    # ── ANONYMIZE tier — soft handling ───────────────────────────────────────

    ask(
        label    = "ANONYMIZE — SSN in request",
        question = (
            "My SSN is 602-76-4532. "
            "Can you explain what a credit score is in one sentence?"
        ),
        expected = "200",
        note     = "US_SSN → anonymized to <US_SSN> before forwarding.",
    )

    ask(
        label    = "ANONYMIZE — name and email in request",
        question = (
            "I am Sarah Johnson, sarah.j@email.com. "
            "Can you explain mortgage interest rates in one sentence?"
        ),
        expected = "200",
        note     = (
            "PERSON + EMAIL_ADDRESS → anonymized to placeholders. "
            "One-word reply guarantees no dates in response."
        ),
    )

    # ── B7: RESPONSE fence — LLM generates PII ─────────────────────────────────────

    ask(
        label    = "RESPONSE BLOCK — LLM generates SSN in sample document",
        question = (
            "Write a sample filled-out loan application form "
            "for a fictional customer. Include all typical fields."
        ),
        expected = "451",
        note     = (
            "LLM fills in a fictional SSN, phone, name, DOB as part of "
            "the form. Presidio catches on the way back → 451."
        ),
    )

    ask(
        label    = "RESPONSE BLOCK — LLM generates name and email",
        question = (
            "Write a sample bank appointment confirmation email. "
            "Make up a realistic customer name and email address."
        ),
        expected = "451",
        note     = "LLM invents PERSON + EMAIL_ADDRESS → caught on response side.",
    )

    # ── Clean pass — proves the fence does not break normal use ─────────────

    ask(
        label    = "Clean response — no PII anywhere",
        question = "In one sentence, what is compound interest?",
        expected = "200",
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 65)
    print("Gateway behaviour summary:")
    print()
    print("  BLOCK     (400) → regulated PII hard-stopped, $0 cost")
    print("  ANONYMIZE (200) → privacy PII replaced, LLM answers normally")
    print("  RESPONSE  (451) → LLM output contained PII, stopped at exit")
    print("  PASS      (200) → clean round trip, no PII anywhere")
    print()
    print("Check traces/trace.jsonl — look for:")
    print("  decision: BLOCKED    → hard stop")
    print("  decision: ANONYMIZED → soft handling, clean_text shows what OpenAI saw")
    print("  decision: FORWARDED  → passed all checks")
    print("=" * 65)
