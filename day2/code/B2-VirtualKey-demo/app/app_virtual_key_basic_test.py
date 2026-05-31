"""
The Company App  (Demo B2 - LiteLLM virtual keys)
=================================================
Calls the LiteLLM gateway with an INTERNAL virtual key. The real OpenAI
key lives only in the gateway.

This version is HARDENED: it retries on connection errors instead of
crashing. When the gateway restarts (deploy, config reload, crash), the
app should hiccup - not fall over. That resilience is the cost of
centralising on a gateway: every app must tolerate the gateway blinking.
"""
import os
import random
import time
import httpx

GATEWAY_URL  = os.environ["GATEWAY_URL"]
INTERNAL_KEY = os.environ.get("INTERNAL_KEY", "").strip()

# Fail fast if the virtual key is missing. Better than a cryptic stack trace.
if not INTERNAL_KEY:
    raise SystemExit(
        "\n" + "=" * 64 + "\n"
        "ERROR: INTERNAL_KEY is not set in the environment.\n"
        "\n"
        "Fix:\n"
        "  1. Mint a virtual key in the LiteLLM UI (http://localhost:4000/ui)\n"
        "  2. Paste the key into .env as INTERNAL_KEY=sk-...\n"
        "  3. Run: docker compose up app --force-recreate\n"
        + "=" * 64
    )

# A small bank of finance questions. Two repeat on purpose - they will show
# cache hits later in Stage 4 (semantic caching). Same app, more guardrails.
QUESTIONS = [
    "In one sentence, what is a bank interest rate?",
    "In one sentence, what is compound interest?",
    "In one sentence, what is a credit score?",
    "In one sentence, what is the prime rate?",
    "In one sentence, what is inflation?",
    "In one sentence, what is a mortgage?",
    "In one sentence, what is a bank interest rate?",   # repeat (cache demo later)
    "In one sentence, what is APR?",
    "In one sentence, what is a 401(k)?",
    "In one sentence, what is compound interest?",      # repeat (cache demo later)
]


def ask(question, max_retries=3):
    """Send one question through the gateway, retrying on connection errors.

    Two kinds of failure are handled very differently:

      - ConnectError (gateway unreachable, e.g. restarting): TRANSIENT.
        Wait with exponential backoff and retry. This is what stops the
        app crashing when LiteLLM restarts.

      - An HTTP response that is not 200 (401 revoked, 429 rate-limited,
        500 vendor error): NOT a connection problem. We do NOT retry these;
        we print the gateway's verdict and move on. Retrying a 401 would
        be pointless - the key is dead until someone re-mints it.
    """
    print(f"\n>>> Asking: {question!r}")
    print(f"    Using internal key: {INTERNAL_KEY[:20]}...")

    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = httpx.post(
                f"{GATEWAY_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {INTERNAL_KEY}"},
                json={
                    # "openai-chat" is the model_name from config.yaml - NOT a
                    # real OpenAI model id. LiteLLM maps it to gpt-4o-mini.
                    "model": "openai-chat",
                    "messages": [{"role": "user", "content": question}],
                },
                timeout=40,
            )
            break  # got an HTTP response (even a 4xx) - stop retrying
        except httpx.ConnectError:
            wait = 2 ** attempt          # 2s, 4s, 8s - exponential backoff
            print(f"    Gateway unreachable (attempt {attempt}/{max_retries}). "
                  f"Retrying in {wait}s...")
            time.sleep(wait)

    if response is None:
        # All retries exhausted - the gateway never came back.
        print("    Gateway still down after retries. Skipping this question.")
        return

    print(f"    HTTP status: {response.status_code}")
    try:
        data = response.json()
    except Exception:
        print(f"    Non-JSON response: {response.text[:200]}")
        return

    if response.status_code == 200:
        answer = data["choices"][0]["message"]["content"]
        print(f"    LLM says: {answer.strip()}")
    else:
        # 401 (revoked), 429 (rate limit / budget), 500 (vendor), etc.
        print(f"    Failed: {data.get('error', data)}")


if __name__ == "__main__":
    print("=" * 64)
    print("Company app - calls the LiteLLM gateway with an INTERNAL key.")
    print("The real OpenAI key lives only in the gateway.")
    print("=" * 64)

    while True:
        question = random.choice(QUESTIONS)
        ask(question)
        #time.sleep(0.1)


        