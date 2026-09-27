"""Vera bot — HTTP surface required by the magicpin judge harness.

Endpoints (challenge-testing-brief.md §2):
    GET  /v1/healthz   liveness + context counts
    GET  /v1/metadata  team identity
    POST /v1/context   versioned context push
    POST /v1/tick      proactive-send decision point
    POST /v1/reply     merchant/customer reply -> send | wait | end

Design rule: the bot must never 500. A malformed request is a 4xx with a
readable body; an internal failure degrades to a valid empty response, because
operational penalties are scored separately from message quality.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import config
from store import (
    SCOPES,
    START_TIME,
    contexts,
    conversations,
    dispatch,
    utc_now_iso,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vera")

app = FastAPI(title="Vera — magicpin AI Challenge", version=config.BOT_VERSION)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


async def _json_body(request: Request) -> Optional[dict]:
    """Parse the request body, tolerating anything that isn't valid JSON."""
    try:
        raw = await request.body()
    except Exception:  # pragma: no cover - defensive
        return None
    if not raw:
        return {}
    if len(raw) > config.CONTEXT_PAYLOAD_CAP_BYTES:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------
# GET /  — human-facing index
# --------------------------------------------------------------------------
# The judge only ever calls /v1/*, but anyone checking the deploy by opening
# the base URL in a browser should see something other than a 404.


@app.get("/")
async def index() -> dict:
    return {
        "service": "Vera — magicpin AI Challenge",
        "version": config.BOT_VERSION,
        "status": "ok",
        "endpoints": {
            "GET  /v1/healthz": "liveness + loaded context counts",
            "GET  /v1/metadata": "team identity",
            "POST /v1/context": "versioned context push",
            "POST /v1/tick": "proactive-send decision point",
            "POST /v1/reply": "reply -> send | wait | end",
            "POST /v1/teardown": "wipe state (optional)",
        },
    }


# --------------------------------------------------------------------------
# GET /v1/healthz
# --------------------------------------------------------------------------


@app.get("/v1/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": contexts.counts(),
    }


# --------------------------------------------------------------------------
# GET /v1/metadata
# --------------------------------------------------------------------------


@app.get("/v1/metadata")
async def metadata() -> dict:
    return config.metadata()


# --------------------------------------------------------------------------
# POST /v1/context
# --------------------------------------------------------------------------


@app.post("/v1/context")
async def push_context(request: Request) -> JSONResponse:
    body = await _json_body(request)
    if body is None:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "malformed_body",
                     "details": "body must be a JSON object within the size cap"},
        )

    scope = str(body.get("scope", "")).strip()
    context_id = str(body.get("context_id", "")).strip()
    version = body.get("version", 0)

    if scope not in SCOPES:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope",
                     "details": f"scope must be one of {list(SCOPES)}"},
        )
    if not context_id:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "missing_context_id",
                     "details": "context_id is required"},
        )
    try:
        version = int(version)
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_version",
                     "details": "version must be an integer"},
        )

    payload = body.get("payload")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_payload",
                     "details": "payload must be an object"},
        )

    accepted, current, reason = contexts.push(
        scope, context_id, version, payload,
        delivered_at=str(body.get("delivered_at", "")),
    )

    if not accepted:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": reason,
                     "current_version": current.version if current else 0},
        )

    return JSONResponse(
        status_code=200,
        content={
            "accepted": True,
            "ack_id": f"ack_{context_id}_v{version}",
            "stored_at": utc_now_iso(),
        },
    )


# --------------------------------------------------------------------------
# POST /v1/tick
# --------------------------------------------------------------------------


@app.post("/v1/tick")
async def tick(request: Request) -> dict:
    body = await _json_body(request) or {}
    dispatch.last_tick_at = str(body.get("now") or utc_now_iso())
    available = body.get("available_triggers") or []
    if not isinstance(available, list):
        available = []

    # Filled in by composer/tick dispatch (phase 3).
    from tick import build_actions

    try:
        actions = build_actions(available, dispatch.last_tick_at)
    except Exception:  # pragma: no cover - never 500 on the judge
        log.exception("tick failed; returning empty actions")
        actions = []

    return {"actions": actions[: config.MAX_ACTIONS_PER_TICK]}


# --------------------------------------------------------------------------
# POST /v1/reply
# --------------------------------------------------------------------------


@app.post("/v1/reply")
async def reply(request: Request) -> dict:
    body = await _json_body(request)
    if body is None:
        return {"action": "end", "rationale": "malformed reply body; nothing safe to send"}

    conversation_id = str(body.get("conversation_id", "")).strip()
    if not conversation_id:
        return {"action": "end", "rationale": "no conversation_id supplied; nothing to reply to"}

    # Filled in by reply handler (phase 4).
    from reply_handler import handle_reply

    try:
        return handle_reply(body)
    except Exception:  # pragma: no cover - never 500 on the judge
        log.exception("reply handling failed; ending conversation")
        return {"action": "end", "rationale": "internal handling error; closing safely"}


# --------------------------------------------------------------------------
# POST /v1/teardown  (optional per brief §11 — wipes state at end of test)
# --------------------------------------------------------------------------


@app.post("/v1/teardown")
async def teardown() -> dict:
    global contexts, conversations, dispatch
    from store import ContextStore, ConversationStore, DispatchState

    contexts = ContextStore()
    conversations = ConversationStore()
    dispatch = DispatchState()

    # Rebind in the modules that imported them by name.
    import store as _store

    _store.contexts = contexts
    _store.conversations = conversations
    _store.dispatch = dispatch

    return {"wiped": True, "at": utc_now_iso()}


# Keep uvicorn happy when this module is the ASGI app.
__all__ = ["app"]
