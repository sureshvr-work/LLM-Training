"""
The Company App
===============
A tiny app that asks an LLM a question.

The ONE important thing: it calls the GATEWAY, not the LLM directly.
Notice there is NO API key in this file - the gateway holds the real key.

It runs inside its own container. To reach the gateway container we use the
Docker Compose service name "gateway" - NOT "localhost". Inside a container,
"localhost" means the container itself.
"""
import time
import httpx

# The app points at the gateway by its Docker service name "gateway".
# Inside Docker Compose, service names resolve as hostnames on the
# internal Docker network.
GATEWAY_URL    = "http://gateway:8000/v1/chat/completions"
GATEWAY_HEALTH = "http://gateway:8000/health"


def wait_for_gateway():
    """Wait until the gateway is actually ready to accept requests.

    docker-compose 'depends_on' only waits for the gateway container to
    START - not for the web server inside it to finish booting. So the app
    must poll the gateway's /health endpoint until it answers.
    """
    for attempt in range(1, 21):           # try for up to ~20 seconds
        try:
            httpx.get(GATEWAY_HEALTH, timeout=2)
            print("Gateway is ready.\n")
            return
        except Exception:
            print(f"Waiting for gateway to be ready... ({attempt})")
            time.sleep(1)
    raise SystemExit("Gateway never came up - giving up.")


def ask(question):
    """Send one question through the gateway and print what comes back."""
    print(f"\n>>> Asking: {question!r}")

    response = httpx.post(
        GATEWAY_URL,
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": question}],
        },
        timeout=40,
    )

    print(f"    HTTP status: {response.status_code}")
    data = response.json()

    if response.status_code == 200:
        # Success - the LLM answered.
        answer = data["choices"][0]["message"]["content"]
        print(f"    LLM says: {answer.strip()}")
    else:
        # The gateway blocked it, or something failed.
        error = data.get("error", {})
        print(f"    Blocked/failed: {error.get('message', data)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Company app - all calls go through the guardrail gateway")
    print("=" * 60)

    # Wait for the gateway before sending anything.
    wait_for_gateway()

    # ------------------------------------------------------------------
    # Test 1: normal question — all rules pass, gateway forwards to LLM.
    # Expected: HTTP 200, LLM replies.
    # ------------------------------------------------------------------
    ask("In one sentence, what is a bank interest rate?")

    ## ------------------------------------------------------------------
    # Test 2: banned keyword — RULE 1 fires, blocked before LLM is called.
    # Expected: HTTP 403, gateway returns error, OpenAI never sees it.
    # ------------------------------------------------------------------
    ask("Tell me about projectfalcon, our new release.")

    # ------------------------------------------------------------------
    # Test 3: huge prompt — RULE 2 fires, blocked before LLM is called.
    # Expected: HTTP 413, gateway returns error, OpenAI never sees it.
    # ------------------------------------------------------------------
    ask("Summarize this: " + ("data " * 3000))

    print("\n" + "=" * 60)
    print("Done.")
    print("Check the gateway console — every request was traced.")
    print("Check traces/trace.jsonl on your host machine.")
    print("Only the 200 request cost tokens. The blocked ones: $0.")
    print("=" * 60)
