"""Runtime configuration.

Everything the bot needs to run is read from environment variables so the same
image works locally, on Railway, and anywhere else. Nothing secret is ever
stored in the repo — see `.env.example`.

The LLM is deliberately optional: until `LLM_PROVIDER` + `LLM_API_KEY` are set,
the bot falls back to the deterministic grounded composer in `composer.py`.
That keeps every endpoint testable and keeps output stable for a given input,
which is what the judge spec requires.
"""

from __future__ import annotations

import os


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


# --------------------------------------------------------------------------
# Identity — surfaced on GET /v1/metadata
# --------------------------------------------------------------------------
TEAM_NAME = _env("TEAM_NAME", "Team Alpha")
TEAM_MEMBERS = [m.strip() for m in _env("TEAM_MEMBERS", "Divyansh").split(",") if m.strip()]
CONTACT_EMAIL = _env("CONTACT_EMAIL", "team@example.com")
BOT_VERSION = _env("BOT_VERSION", "0.1.0")

# --------------------------------------------------------------------------
# LLM (optional)
# --------------------------------------------------------------------------
# One of: openai | anthropic | gemini | deepseek | groq | openrouter
LLM_PROVIDER = _env("LLM_PROVIDER", "").lower()
LLM_API_KEY = _env("LLM_API_KEY", "")
LLM_MODEL = _env("LLM_MODEL", "")

# Determinism is a hard requirement of the challenge: identical inputs must
# produce identical messages. 0.0 is the only correct value here.
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.0)

# The judge allows 30s per call. Composition happens inside /v1/tick and
# /v1/reply, so the LLM budget is deliberately much smaller than the ceiling.
LLM_TIMEOUT_S = _env_float("LLM_TIMEOUT_S", 12.0)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 700)


def llm_configured() -> bool:
    """True when a real LLM backend is available."""
    return bool(LLM_PROVIDER and LLM_API_KEY)


# --------------------------------------------------------------------------
# Judge-harness limits (challenge-testing-brief.md §5)
# --------------------------------------------------------------------------
MAX_ACTIONS_PER_TICK = _env_int("MAX_ACTIONS_PER_TICK", 20)
CONTEXT_PAYLOAD_CAP_BYTES = _env_int("CONTEXT_PAYLOAD_CAP_BYTES", 500 * 1024)

# Composer: how much of the category knowledge pack to inline into a prompt.
MAX_DIGEST_ITEMS = _env_int("MAX_DIGEST_ITEMS", 6)
MAX_OFFERS = _env_int("MAX_OFFERS", 6)

# If a merchant has not replied to this many consecutive nudges, stop nudging.
MAX_UNANSWERED_NUDGES = _env_int("MAX_UNANSWERED_NUDGES", 3)

# Auto-reply thresholds. Production Vera wastes turns here; we do not.
AUTO_REPLY_FLAG_AFTER = _env_int("AUTO_REPLY_FLAG_AFTER", 1)  # try once
AUTO_REPLY_WAIT_AFTER = _env_int("AUTO_REPLY_WAIT_AFTER", 2)  # then back off
AUTO_REPLY_END_AFTER = _env_int("AUTO_REPLY_END_AFTER", 3)  # then stop

# How long to back off after an auto-reply, in seconds.
AUTO_REPLY_WAIT_SECONDS = _env_int("AUTO_REPLY_WAIT_SECONDS", 14400)  # 4h


def metadata() -> dict:
    """Payload for GET /v1/metadata."""
    if llm_configured():
        approach = (
            "Grounded signal selection over the 4 contexts, then a deterministic "
            "composer per trigger kind with post-generation validation"
        )
        model = f"{LLM_PROVIDER}:{LLM_MODEL or 'default'}"
    else:
        approach = (
            "Grounded signal selection over the 4 contexts, then a deterministic "
            "template composer per trigger kind (no LLM required)"
        )
        model = "deterministic-composer"
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": model,
        "approach": approach,
        "contact_email": CONTACT_EMAIL,
        "version": BOT_VERSION,
        "submitted_at": _env("SUBMITTED_AT", ""),
    }
