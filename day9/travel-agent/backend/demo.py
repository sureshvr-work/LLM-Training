"""
demo.py — run the loop once with the mock engine and print the turns.

No server, no keys, no network. The fastest way to see the loop work:

    cd backend && python demo.py
"""
from providers.base import get_provider
from tools.mock_directories import build_registry
from loop import run_loop

GOAL = "Goa · round-trip + 5★ hotel · 5 nights · under $5,000"

if __name__ == "__main__":
    turns = run_loop(GOAL, get_provider("mock"), build_registry())
    for t in turns:
        print(f"\n── turn {t.n} ──")
        print("REASON:", t.thought)
        for c in t.calls:
            print("  ACT    →", c["name"], c["arguments"])
        for r in t.results:
            print("  RESULT ←", r["name"], r["content"])
        if t.final:
            print("FINAL:", t.final)
