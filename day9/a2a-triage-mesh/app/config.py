"""
config.py — every tunable in one place, read from the environment.

The SAME image runs as any agent in the mesh; AGENT_ROLE picks which one. In
docker-compose each service sets its own AGENT_ROLE, SELF_URL, and (for the
router) PEERS — the list of specialist base URLs it will discover.

Keys NEVER live in code. The default `mock` engine needs none, so the whole
mesh comes up with `docker compose up` and no .env.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _csv(v: str) -> list[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


class Config:
    # which agent this process is: "router" | "cardiology" | "pulmonology" |
    # "dermatology" | "neurology"
    ROLE = os.getenv("AGENT_ROLE", "router").lower()

    # this agent's own externally-reachable base (used to fill its card's url).
    # In docker this is the service name, e.g. http://cardiology:8000
    SELF_URL = os.getenv("SELF_URL", "").rstrip("/")

    # ROUTER ONLY: the specialist base URLs to discover (comma-separated).
    PEERS = _csv(os.getenv("PEERS",
                 "http://127.0.0.1:8101,http://127.0.0.1:8102,"
                 "http://127.0.0.1:8103,http://127.0.0.1:8104"))

    # SPECIALIST ONLY: does this agent advertise streaming in its card? Set
    # STREAMING=false on a container to prove the router falls back to
    # message/send for it (capability-driven transport).
    STREAMING = os.getenv("STREAMING", "true").lower() not in ("false", "0", "no")

    # ── LLM engines (REASON: router extraction + each specialist's loop) ──
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # ── resilience / pacing ──
    HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "12"))
    MAX_TURNS = int(os.getenv("MAX_TURNS", "6"))
    STEP_DELAY = float(os.getenv("STEP_DELAY", "0.45"))   # demo pacing


cfg = Config()
