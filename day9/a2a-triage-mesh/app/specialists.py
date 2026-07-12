"""
specialists.py — the four ward agents, each of which runs as its OWN container.

Each spec is the same handful of facts the single-process sample used — a system
prompt, a SCOPED tool set, and the (system, acuity) tuples it claims — but here
each one is published as an A2A Agent Card and served by its own service. The
router discovers these cards; that is the whole point of going multi-container.

Deliberate gaps (so discovery/refusal is visible): neuro·LOW and derm·HIGH are
NOT claimed by anyone, and gastro/ortho/general have no agent at all. A case that
maps there is refused — you can see the missing edge in the discovered graph.
"""
import asyncio

import a2a
from config import cfg
from engine import Decision
from tools import specs as tool_specs, call as tool_call


SPECS = {
    "cardiology": {
        "title": "Cardiology", "ward": "Cardiology · Cath Lab",
        "system": (
            "You are the Cardiology ward agent in a schematic triage simulation (not real "
            "medical software). A cardiac case has been routed to you. Use ONLY your tools "
            "to work it up: read vitals, order an ECG, check the cath lab, page on-call if "
            "the picture is high-acuity, and consult the protocol. Call ONE tool at a time. "
            "When you have enough, stop and write a short disposition grounded only in what "
            "the tools returned. Keep it operational; give no medical advice."),
        "tools": ["get_vitals", "order_ecg", "check_cath_lab", "page_oncall", "lookup_protocol"],
        "claims": [("cardiac", "HIGH"), ("cardiac", "LOW")],
        "skill": {"id": "cardiac-workup", "name": "Cardiac workup",
                  "description": "Triage and work up cardiac presentations (ECG, cath lab, on-call).",
                  "tags": ["cardiac", "HIGH", "LOW", "ecg", "cath-lab"],
                  "examples": ["chest tightness with left-arm pain", "palpitations, sweating"]},
    },
    "pulmonology": {
        "title": "Pulmonology", "ward": "Pulmonology · ICU",
        "system": (
            "You are the Pulmonology ward agent in a schematic triage simulation (not real "
            "medical software). A respiratory case has been routed to you. Use ONLY your "
            "tools: read vitals, order labs (e.g. ABG), order chest imaging, check ICU beds, "
            "page on-call for high-acuity, and consult the protocol. One tool at a time. "
            "Finish with a short disposition grounded only in tool results. No medical advice."),
        "tools": ["get_vitals", "order_labs", "order_imaging", "check_beds", "page_oncall", "lookup_protocol"],
        "claims": [("respiratory", "HIGH"), ("respiratory", "LOW")],
        "skill": {"id": "respiratory-workup", "name": "Respiratory workup",
                  "description": "Triage and work up respiratory presentations (ABG, imaging, ICU).",
                  "tags": ["respiratory", "HIGH", "LOW", "abg", "icu"],
                  "examples": ["short of breath, wheezing badly", "productive cough, low sats"]},
    },
    "dermatology": {
        "title": "Dermatology", "ward": "Dermatology · Clinic",
        "system": (
            "You are the Dermatology agent in a schematic triage simulation (not real medical "
            "software). A low-acuity skin case has been routed to you. Use ONLY your tools: "
            "consult the protocol, schedule dermoscopy/imaging if useful, and book an "
            "outpatient clinic slot. One tool at a time. Finish with a short disposition "
            "grounded only in tool results. No medical advice."),
        "tools": ["book_clinic", "order_imaging", "lookup_protocol"],
        "claims": [("dermatology", "LOW")],
        "skill": {"id": "derm-clinic", "name": "Dermatology clinic",
                  "description": "Handle low-acuity skin cases (protocol, dermoscopy, clinic booking).",
                  "tags": ["dermatology", "LOW", "clinic"],
                  "examples": ["itchy red rash on the elbow", "a changing mole, non-urgent"]},
    },
    "neurology": {
        "title": "Neurology", "ward": "Neurology · Stroke Unit",
        "system": (
            "You are the Neurology ward agent in a schematic triage simulation (not real "
            "medical software). A high-acuity neuro case has been routed to you. Use ONLY "
            "your tools: read vitals, order head imaging (e.g. CT head), check stroke-unit "
            "beds, page on-call, and consult the protocol. One tool at a time. Finish with a "
            "short disposition grounded only in tool results. No medical advice."),
        "tools": ["get_vitals", "order_imaging", "page_oncall", "check_beds", "lookup_protocol"],
        "claims": [("neuro", "HIGH")],
        "skill": {"id": "neuro-stroke", "name": "Neuro / stroke workup",
                  "description": "Work up high-acuity neuro presentations (CT head, stroke unit, on-call).",
                  "tags": ["neuro", "HIGH", "stroke-unit"],
                  "examples": ["sudden severe headache, slurred speech", "facial droop, weakness"]},
    },
}


def get_spec(role: str) -> dict | None:
    return SPECS.get(role)


def build_card(role: str, base_url: str, streaming: bool | None = None) -> dict:
    """The Agent Card this specialist serves at /.well-known/agent-card.json."""
    s = SPECS[role]
    streams = cfg.STREAMING if streaming is None else streaming
    return {
        "name": f"{s['title']} Agent",
        "description": f"{s['ward']} — {s['skill']['description']}",
        "version": "1.0.0",
        "url": f"{base_url}/a2a",
        "capabilities": {"streaming": bool(streams), "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [s["skill"]],
        # ── discovery contract for this demo: what the agent claims + its scope ──
        "x-triage-claims": [list(c) for c in s["claims"]],
        "x-ward": s["ward"],
        "x-tools": s["tools"],
    }


# ── the scoped ReAct loop, streamed as A2A task updates ──────────────────────
async def _run_loop(role: str, case: str, provider, task_id, context_id, fail: bool = False):
    """Async generator of A2A streaming `result` payloads for this specialist's work."""
    s = SPECS[role]
    history = [{"role": "system", "content": s["system"]},
               {"role": "user", "content": case}]
    d = cfg.STEP_DELAY

    yield a2a.status_event(task_id, context_id, "submitted", False)
    await asyncio.sleep(d)
    yield a2a.status_event(task_id, context_id, "working", False)

    for turn in range(1, cfg.MAX_TURNS + 1):
        try:
            decision = await asyncio.to_thread(provider.reason, history, tool_specs(only=s["tools"]))
        except Exception as e:                               # noqa: BLE001
            yield {"taskId": task_id, "contextId": context_id, "kind": "agent-error",
                   "message": f"{type(e).__name__}: {e}"}
            yield a2a.status_event(task_id, context_id, "failed", True)
            return

        history.append({"role": "assistant", "thought": decision.thought,
                        "tool": ({"id": decision.tool_call.id, "name": decision.tool_call.name,
                                  "args": decision.tool_call.args} if decision.tool_call else None)})
        # progress event: the model's reasoning this turn (demo-level visibility)
        yield {"taskId": task_id, "contextId": context_id, "kind": "agent-reason",
               "turn": turn, "thought": decision.thought,
               "tool": ({"name": decision.tool_call.name, "args": decision.tool_call.args}
                        if decision.tool_call else None)}
        await asyncio.sleep(d)

        if not decision.tool_call:
            artifact = a2a.text_artifact(decision.final or decision.thought, name="disposition")
            yield a2a.artifact_event(task_id, context_id, artifact)
            await asyncio.sleep(d * 0.6)
            yield a2a.status_event(task_id, context_id, "completed", True)
            return

        tc = decision.tool_call
        # ── failure injection: the first tool throws, the task FAILS ──
        if fail:
            yield {"taskId": task_id, "contextId": context_id, "kind": "agent-tool",
                   "turn": turn, "name": tc.name, "ok": False,
                   "result": {"error": f"{tc.name} raised: upstream system unavailable (injected)"}}
            await asyncio.sleep(d)
            yield {"taskId": task_id, "contextId": context_id, "kind": "agent-error",
                   "message": f"tool '{tc.name}' failed: upstream system unavailable (injected)"}
            yield a2a.status_event(task_id, context_id, "failed", True)
            return

        result = await asyncio.to_thread(tool_call, tc.name, tc.args, s["tools"])
        ok = not (isinstance(result, dict) and "error" in result)
        history.append({"role": "tool", "id": tc.id, "name": tc.name, "content": str(result)[:2000]})
        yield {"taskId": task_id, "contextId": context_id, "kind": "agent-tool",
               "turn": turn, "name": tc.name, "ok": ok, "result": result}
        await asyncio.sleep(d)

    artifact = a2a.text_artifact("Reached the turn limit — partial disposition.", name="disposition")
    yield a2a.artifact_event(task_id, context_id, artifact)
    yield a2a.status_event(task_id, context_id, "completed", True)


async def handle_stream(role: str, req: dict, provider):
    """message/stream → SSE frames (real A2A streaming response)."""
    req_id = req.get("id")
    params = req.get("params") or {}
    meta = params.get("metadata") or {}
    fail = bool(meta.get("fail"))
    case = a2a.first_text(params.get("message") or {})
    context_id = params.get("contextId") or a2a.new_id("ctx")
    task_id = a2a.new_id("task")
    async for result in _run_loop(role, case, provider, task_id, context_id, fail=fail):
        yield a2a.sse(a2a.rpc_result(req_id, result))


def handle_send(role: str, req: dict, provider) -> dict:
    """message/send → one completed (or failed) Task (synchronous form, for curl)."""
    s = SPECS[role]
    params = req.get("params") or {}
    meta = params.get("metadata") or {}
    fail = bool(meta.get("fail"))
    case = a2a.first_text(params.get("message") or {})
    context_id = params.get("contextId") or a2a.new_id("ctx")
    task_id = a2a.new_id("task")
    history = [{"role": "system", "content": s["system"]}, {"role": "user", "content": case}]
    final = "no disposition"
    for _turn in range(cfg.MAX_TURNS):
        decision = provider.reason(history, tool_specs(only=s["tools"]))
        history.append({"role": "assistant", "thought": decision.thought,
                        "tool": ({"id": decision.tool_call.id, "name": decision.tool_call.name,
                                  "args": decision.tool_call.args} if decision.tool_call else None)})
        if not decision.tool_call:
            final = decision.final or decision.thought
            break
        tc = decision.tool_call
        if fail:
            task = a2a.make_task(task_id, context_id, "failed", artifacts=[
                a2a.text_artifact(f"tool '{tc.name}' failed: upstream system unavailable (injected)",
                                  name="error")])
            return a2a.rpc_result(req.get("id"), task)
        result = tool_call(tc.name, tc.args, s["tools"])
        history.append({"role": "tool", "id": tc.id, "name": tc.name, "content": str(result)[:2000]})
    artifact = a2a.text_artifact(final, name="disposition")
    task = a2a.make_task(task_id, context_id, "completed", artifacts=[artifact])
    return a2a.rpc_result(req.get("id"), task)
