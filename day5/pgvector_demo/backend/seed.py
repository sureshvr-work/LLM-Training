"""
seed.py — a tiny mixed corpus (finance + healthcare + general).

It's deliberately written so that EXACT keyword search misses things that VECTOR
search finds by meaning. For example:
  query "heart attack"      -> no row literally contains those words,
                               but the myocardial-infarction row is a near vector.
  query "money laundering"  -> the row says "anti-money-laundering" (hyphen),
                               so ILIKE '%money laundering%' misses; vectors don't.
"""
DOCS = [
    ("AML threshold",        "A suspicious activity report must be filed for cash transactions over $10,000."),
    ("Money laundering",     "Anti-money-laundering rules require banks to report large or unusual transfers."),
    ("KYC onboarding",       "Know Your Customer checks verify a client's identity when an account is opened."),
    ("Wire review",          "Wire transfers flagged by the system are escalated to the compliance team."),
    ("Card APR",             "The annual percentage rate on this credit card is 36% after the introductory period."),
    ("Heart attack",         "A myocardial infarction happens when blood flow to the cardiac muscle is blocked."),
    ("Chest pain triage",    "Patients arriving with chest discomfort should be assessed for cardiac causes right away."),
    ("Prior authorization",  "The insurer requires pre-approval before it will cover the prescribed procedure."),
    ("Diabetes care",        "Keeping blood glucose stable helps people avoid long-term complications."),
    ("Refill reminder",      "The pharmacy sends a notification when a prescription is due to be refilled."),
    ("Account access",       "Users can regain entry to their account through the forgotten-password link."),
    ("Returns policy",       "Shoppers may send an item back within thirty days of the purchase date."),
    ("Service outage",       "The site was briefly unavailable because of a database connection failure."),
    ("Morning coffee",       "Lots of people enjoy a warm cup of coffee to begin their working day."),
]

# Suggested demo queries -> the UI offers these as one-click chips.
EXAMPLE_QUERIES = ["heart attack", "money laundering", "can't log in", "send a product back"]
