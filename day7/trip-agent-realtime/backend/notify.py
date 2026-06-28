"""
notify.py — fires once, after the plan is ready: emails it via Resend.

Best-effort by design: a notification failure must never break the agent run,
so callers get back a result dict instead of a raised exception.
"""
from http_client import request, HttpError
from config import cfg


def send_email(to: str, subject: str, body: str):
    # retries=0: sending is not idempotent — a retried POST risks a duplicate email.
    return request("POST", "https://api.resend.com/emails", retries=0,
                   headers={"Authorization": f"Bearer {cfg.RESEND_API_KEY}"},
                   json={"from": cfg.NOTIFY_FROM, "to": [to],
                         "subject": subject, "text": body})


def notify_plan_ready(goal: str, plan_text: str, to: str = "") -> dict:
    to = to or cfg.NOTIFY_EMAIL_TO
    if not to:
        return {}
    try:
        send_email(to, f"Trip plan ready: {goal}", plan_text)
        return {"email": {"ok": True}}
    except HttpError as e:
        return {"email": {"ok": False, "error": str(e)}}
