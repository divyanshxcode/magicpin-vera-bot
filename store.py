"""In-memory state for the Vera bot.

The judge harness pushes context over time and expects the bot to *use* the
latest version of everything, including context that arrives mid-conversation.
The judge never restarts the bot during a run, so an in-process store is
sufficient and keeps the deploy trivial.

Three concerns live here:

1. `ContextStore`   — versioned, idempotent context keyed by (scope, context_id)
2. `ConversationStore` — per-conversation turn history and lifecycle state
3. `DispatchState`  — cross-conversation bookkeeping (suppression, opt-outs)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

SCOPES = ("category", "merchant", "customer", "trigger")

START_TIME = time.time()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------


@dataclass
class ContextEntry:
    scope: str
    context_id: str
    version: int
    payload: dict
    delivered_at: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.scope, self.context_id)


class ContextStore:
    """Versioned context keyed by (scope, context_id).

    Contract from challenge-testing-brief.md §2.1:
      - idempotent by (context_id, version); re-pushing the same version is a no-op
      - a higher version replaces the prior one atomically
      - a lower/equal version is rejected as stale
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], ContextEntry] = {}
        self._lock = threading.RLock()

    def reset(self) -> None:
        """Drop all stored context. Tests only — see ConversationStore.reset."""
        with self._lock:
            self._data.clear()

    def push(
        self, scope: str, context_id: str, version: int, payload: dict, delivered_at: str = ""
    ) -> tuple[bool, Optional[ContextEntry], str]:
        """Store a context version.

        Returns (accepted, current_entry, reason).

        Version rules, per the challenge brief:
          - re-posting the *same* (context_id, version) is idempotent and
            succeeds — a no-op that still acks, so a judge that pushes the
            opening context again never sees a failure
          - only a strictly *lower* version is a conflict
          - a higher version replaces the prior one atomically
        """
        with self._lock:
            key = (scope, context_id)
            existing = self._data.get(key)
            if existing is not None and existing.version > version:
                return False, existing, "stale_version"
            if existing is not None and existing.version == version:
                return True, existing, "duplicate_version"
            self._data[key] = ContextEntry(
                scope=scope,
                context_id=context_id,
                version=version,
                payload=payload or {},
                delivered_at=delivered_at,
            )
            return True, self._data[key], ""

    def get(self, scope: str, context_id: str) -> Optional[dict]:
        with self._lock:
            entry = self._data.get((scope, context_id))
            return entry.payload if entry else None

    def get_entry(self, scope: str, context_id: str) -> Optional[ContextEntry]:
        with self._lock:
            return self._data.get((scope, context_id))

    def version(self, scope: str, context_id: str) -> int:
        with self._lock:
            entry = self._data.get((scope, context_id))
            return entry.version if entry else 0

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in SCOPES}
        with self._lock:
            for (scope, _cid) in self._data:
                if scope in out:
                    out[scope] += 1
        return out

    def all_of_scope(self, scope: str) -> list[dict]:
        with self._lock:
            return [e.payload for (s, _c), e in self._data.items() if s == scope]

    def total(self) -> int:
        with self._lock:
            return len(self._data)


# --------------------------------------------------------------------------
# Conversations
# --------------------------------------------------------------------------

# Lifecycle states
ACTIVE = "active"
ENDED = "ended"          # merchant opted out, or bot closed the conversation
SUPPRESSED = "suppressed"  # do not message this merchant again for now


@dataclass
class Turn:
    turn_number: int
    from_role: str
    message: str
    received_at: str = ""
    action: str = ""          # what the bot did with this turn
    is_auto_reply: bool = False
    intent: str = ""          # classified intent (commit / question / hostile / ...)


@dataclass
class Conversation:
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    trigger_id: Optional[str] = None
    send_as: str = "vera"
    state: str = ACTIVE
    created_at: str = field(default_factory=utc_now_iso)

    # Bot's own sent messages, for anti-repetition and continuity.
    sent_bodies: list[str] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)

    # Merchant-received messages we have not had a human answer to.
    unanswered_nudges: int = 0

    # Auto-reply tracking
    auto_reply_streak: int = 0
    last_inbound_text: str = ""

    # What the merchant last said they wanted, so we can honour it.
    last_intent: str = ""

    def has_sent(self, body: str) -> bool:
        return body.strip() in {b.strip() for b in self.sent_bodies}

    def record_send(self, body: str) -> None:
        self.sent_bodies.append(body)

    def is_open(self) -> bool:
        return self.state == ACTIVE

    def recent_inbound(self, n: int = 3) -> list[Turn]:
        inbound = [t for t in self.turns if t.from_role != "vera"]
        return inbound[-n:]

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "merchant_id": self.merchant_id,
            "customer_id": self.customer_id,
            "trigger_id": self.trigger_id,
            "send_as": self.send_as,
            "state": self.state,
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "from": t.from_role,
                    "message": t.message,
                    "received_at": t.received_at,
                    "action": t.action,
                    "is_auto_reply": t.is_auto_reply,
                    "intent": t.intent,
                }
                for t in self.turns
            ],
        }


class ConversationStore:
    def __init__(self) -> None:
        self._data: dict[str, Conversation] = {}
        self._lock = threading.RLock()

    def reset(self) -> None:
        """Drop all conversation state.

        Only the test suites call this. The judge never restarts the bot
        mid-test, so there is no production path that needs this — it exists
        so a test can assert determinism against genuinely fresh state instead
        of state left behind by the previous test.
        """
        with self._lock:
            self._data.clear()

    def create(
        self,
        conversation_id: str,
        merchant_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        trigger_id: Optional[str] = None,
        send_as: str = "vera",
    ) -> Conversation:
        with self._lock:
            conv = self._data.get(conversation_id)
            if conv is None:
                conv = Conversation(
                    conversation_id=conversation_id,
                    merchant_id=merchant_id,
                    customer_id=customer_id,
                    trigger_id=trigger_id,
                    send_as=send_as,
                )
                self._data[conversation_id] = conv
            return conv

    def get(self, conversation_id: str) -> Optional[Conversation]:
        with self._lock:
            return self._data.get(conversation_id)

    def all(self) -> Iterable[Conversation]:
        with self._lock:
            return list(self._data.values())

    def set_state(self, conversation_id: str, state: str) -> None:
        with self._lock:
            conv = self._data.get(conversation_id)
            if conv is not None:
                conv.state = state

    def count(self) -> int:
        with self._lock:
            return len(self._data)


# --------------------------------------------------------------------------
# Cross-conversation dispatch bookkeeping
# --------------------------------------------------------------------------


@dataclass
class DispatchState:
    """Bot-wide memory: what we've already sent, and who has opted out.

    Restraint is explicitly rewarded by the judge, so the bot tracks its own
    send history and declines to re-message when nothing new is worth saying.
    """

    # suppression_key -> last simulated time we sent it
    sent_suppressions: dict[str, str] = field(default_factory=dict)
    # merchant_id -> last simulated time we sent anything at all
    last_sent_to: dict[str, str] = field(default_factory=dict)
    # merchant_id -> last simulated time
    last_tick_at: str = ""
    # merchants who said stop; never message again
    opted_out: set[str] = field(default_factory=set)
    # conversation_ids the judge has ended
    closed_conversations: set[str] = field(default_factory=set)

    def reset(self) -> None:
        """Forget everything we have sent. Tests only — see ConversationStore.reset."""
        self.sent_suppressions.clear()
        self.last_sent_to.clear()
        self.opted_out.clear()
        self.closed_conversations.clear()
        self.last_tick_at = ""


contexts = ContextStore()
conversations = ConversationStore()
dispatch = DispatchState()
