"""
loop.py — the agent loop. This is the whole "agent": sense -> reason -> act,
repeated until done. It is deliberately tiny and engine-agnostic.

    SENSE   assemble / observe the context  (read memory, fold in last result)
    REASON  ask the chosen engine what to do next   (provider.reason)
    ACT     run the tool(s) the engine chose, append results to memory

...until the engine returns a final answer (or we hit a turn cap as a safety net).

The loop returns a list of `Turn`s — one per pass — which is exactly what the UI
renders.
"""
from dataclasses import dataclass
from typing import Optional

from schema import Message, Decision, ToolResult
from memory import ShortTermMemory
from providers.base import Provider
from tools.registry import ToolRegistry


@dataclass
class Turn:
    """One pass of the loop — the unit the UI shows as a 'turn'."""
    n: int
    thought: str                 # REASON: the engine's reasoning
    calls: list                  # ACT: [{"name":..., "arguments":...}]
    results: list                # [{"name":..., "content":...}]
    final: Optional[str] = None  # set on the last turn


def run_loop(goal: str, provider: Provider, registry: ToolRegistry,
             max_turns: int = 8) -> list[Turn]:
    mem = ShortTermMemory()

    # ── SENSE (bootstrap): seed the context with the goal + what we remember ──
    mem.add(Message("system", "You are a travel-planning agent. Book within budget."))
    mem.add(Message("user", goal))
    profile = registry.execute_by_name("get_user_profile")        # read memory
    mem.add(Message("tool", str(profile.content), name="get_user_profile"))

    turns: list[Turn] = []
    for n in range(1, max_turns + 1):
        # ── REASON: the only step that depends on the engine ──
        decision: Decision = provider.reason(mem.history(), registry.specs())
        mem.add(Message("assistant", decision.thought))

        # ── ACT: run each chosen tool, fold its result back into the context ──
        results = []
        for call in decision.tool_calls:
            result: ToolResult = registry.execute(call)
            mem.add(Message("tool", str(result.content), name=result.name))
            results.append({"name": result.name, "content": result.content})

        turns.append(Turn(
            n=n,
            thought=decision.thought,
            calls=[{"name": c.name, "arguments": c.arguments}
                   for c in decision.tool_calls],
            results=results,
            final=decision.final,
        ))

        # ── stop when the engine says it's finished ──
        if decision.done:
            mem.add(Message("assistant", decision.final))
            break

    return turns
