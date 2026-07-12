"""
tools.py — the tool contract + a name->function registry, with PER-AGENT scope,
plus the specialists' clinical tool stubs.

A tool = a function + a JSON-schema spec. `specs(only=…)` hands an agent ONLY
the tools it is allowed to use; `call(name, …, allowed=…)` refuses anything
outside that set at call time. Least privilege, enforced — the Dermatology agent
never even sees order_ecg, and couldn't run it if it tried.

The clinical tools are SCHEMATIC, deterministic stubs — canned data, no real
systems, not medical software and not medical advice.
"""
try:
    from jsonschema import validate, ValidationError
    _HAVE_JSONSCHEMA = True
except Exception:                                        # keep deps optional
    _HAVE_JSONSCHEMA = False

TOOLS: dict[str, dict] = {}


def tool(name, description, schema):
    def decorate(fn):
        TOOLS[name] = {"fn": fn, "spec": {"name": name, "description": description,
                                          "parameters": schema}}
        return fn
    return decorate


def specs(only=None):
    return [t["spec"] for n, t in TOOLS.items() if only is None or n in only]


def call(name, args, allowed=None):
    if allowed is not None and name not in allowed:
        return {"error": f"tool '{name}' is out of this agent's scope"}
    if name not in TOOLS:
        return {"error": f"unknown tool: {name}"}
    if _HAVE_JSONSCHEMA:
        try:
            validate(args, TOOLS[name]["spec"]["parameters"])
        except ValidationError as e:
            return {"error": f"invalid arguments for {name}: {e.message}"}
    return TOOLS[name]["fn"](**args)


# ── the clinical tool stubs (canned, deterministic) ──────────────────────────
@tool("get_vitals", "Fetch the patient's current vital signs.",
      {"type": "object", "properties": {"patient_id": {"type": "string"}},
       "required": ["patient_id"]})
def get_vitals(patient_id):
    return {"patient_id": patient_id, "hr": 104, "bp": "148/92",
            "spo2": 94, "temp_c": 37.1, "note": "schematic demo values"}


@tool("order_ecg", "Order a 12-lead ECG and return its summary.",
      {"type": "object", "properties": {}})
def order_ecg():
    return {"study": "12-lead ECG", "summary": "ST elevation, inferior leads",
            "status": "resulted", "note": "schematic demo value"}


@tool("check_cath_lab", "Check catheterisation-lab availability.",
      {"type": "object", "properties": {}})
def check_cath_lab():
    return {"cath_lab": "Lab 2", "available": True, "ready_in_min": 8}


@tool("order_labs", "Order a lab panel.",
      {"type": "object", "properties": {"panel": {"type": "string"}}, "required": ["panel"]})
def order_labs(panel):
    return {"panel": panel, "status": "ordered", "eta_min": 20}


@tool("order_imaging", "Schedule an imaging study.",
      {"type": "object", "properties": {"kind": {"type": "string"}}, "required": ["kind"]})
def order_imaging(kind):
    return {"kind": kind, "status": "scheduled", "eta_min": 30}


@tool("check_beds", "Check free beds in a ward.",
      {"type": "object", "properties": {"ward": {"type": "string"}}, "required": ["ward"]})
def check_beds(ward):
    return {"ward": ward, "free_beds": 3}


@tool("page_oncall", "Page the on-call clinician for a service.",
      {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]})
def page_oncall(service):
    return {"paged": service, "acknowledged": True, "eta_min": 5}


@tool("book_clinic", "Book an outpatient clinic slot.",
      {"type": "object", "properties": {"dept": {"type": "string"}}, "required": ["dept"]})
def book_clinic(dept):
    return {"dept": dept, "slot": "Tue 10:00", "status": "booked"}


@tool("lookup_protocol", "Look up a care-pathway protocol summary.",
      {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]})
def lookup_protocol(topic):
    return {"topic": topic, "summary": f"[schematic] standard care pathway for '{topic}'."}
