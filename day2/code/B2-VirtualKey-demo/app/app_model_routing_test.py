"""
The Company App  (Demo B2 - LiteLLM virtual keys + model routing)
=================================================
Calls the LiteLLM gateway with an INTERNAL virtual key. The real OpenAI
key lives only in the gateway.

Model routing demo: same app, same key, different models.
Alice's key only allows openai-chat → anthropic-chat gets 403.
"""
import os
import random
import time
import httpx

GATEWAY_URL  = os.environ["GATEWAY_URL"]
INTERNAL_KEY = os.environ.get("INTERNAL_KEY", "").strip()

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

# Each entry is (question, model_name)
# Alice's key only allows openai-chat
# anthropic-chat and google-chat will get 403 → CHECK ② firing live
QUESTIONS = [
    ("In one sentence, what is a bank interest rate?",  "openai-chat"),
    ("In one sentence, what is compound interest?",     "anthropic-chat"),
    ("In one sentence, what is a credit score?",        "google-chat"),
    ("In one sentence, what is the prime rate?",        "openai-chat"),
    ("In one sentence, what is inflation?",             "anthropic-chat"),
    ("In one sentence, what is a mortgage?",            "openai-chat"),
]


def ask(question, model, max_retries=3):
    """Send one question through the gateway using the specified model."""
    print(f"\n>>> Asking : {question!r}")
    print(f"    Model   : {model}")
    print(f"    Key     : {INTERNAL_KEY[:20]}...")

    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = httpx.post(
                f"{GATEWAY_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {INTERNAL_KEY}"},
                json={
                    "model":    model,
                    "messages": [{"role": "user", "content": question}],
                },
                timeout=40,
            )
            break
        except httpx.ConnectError:
            wait = 2 ** attempt
            print(f"    Gateway unreachable (attempt {attempt}/{max_retries}). "
                  f"Retrying in {wait}s...")
            time.sleep(wait)

    if response is None:
        print("    Gateway still down after retries. Skipping.")
        return

    print(f"    HTTP status : {response.status_code}")
    try:
        data = response.json()
    except Exception:
        print(f"    Non-JSON response: {response.text[:200]}")
        return

    if response.status_code == 200:
        answer = data["choices"][0]["message"]["content"]
        print(f"    LLM says    : {answer.strip()}")
    elif response.status_code == 403:
        err = data.get("error", data)
        print(f"    BLOCKED 403 : model '{model}' not allowed for this key")
        print(f"    CHECK ②     : model access control fired — $0 cost")
    else:
        print(f"    Failed {response.status_code} : {data.get('error', data)}")


if __name__ == "__main__":
    print("=" * 64)
    print("Company app — B2 model routing demo")
    print("Alice key: openai-chat only")
    print("anthropic-chat and google-chat → expect 403")
    print("=" * 64)

    while True:
        question, model = random.choice(QUESTIONS)
        ask(question, model)
        time.sleep(2)