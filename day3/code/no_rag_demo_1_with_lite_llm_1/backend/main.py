"""Thin controller: validate, delegate to services, return.
One /api/chat endpoint for every model."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import services

app = FastAPI()


class LoginIn(BaseModel):
    username: str


class ChatIn(BaseModel):
    model: str
    messages: list
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    key: str | None = None


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/login")
async def login(body: LoginIn):
    try:
        key = await services.issue_key(body.username)
    except httpx.HTTPError as e:
        raise HTTPException(500, f"key generation failed: {e}")
    return {"username": body.username, "key": key}


@app.post("/api/chat")
async def chat(body: ChatIn):
    try:
        return await services.complete(
            body.model, body.messages, body.temperature,
            body.max_tokens, body.top_p, body.key,
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)
    except httpx.HTTPError as e:
        raise HTTPException(502, str(e))
