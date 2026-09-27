"""Tick dispatch: decide what (if anything) to proactively send.

`/v1/tick` fires every 5 simulated minutes across a 60-minute window, so a bot
that sends whenever a trigger is available will message a merchant a dozen
times per run. The judge is explicit that this is punished:

    "Can my bot refuse to send when nothing's worth saying?"
    "Yes - return {\"actions\": []}. Restraint is rewarded; spam is penalized."

So every candidate send passes a gauntlet of guards before it ships.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import config
import grounding as g
from composer import compose
from store import contexts, conversations, dispatch as _global_dispatch, utc_now_iso

# Minimum simulated minutes between two sends to the same merchant. Three
# messages in an hour is exactly the pattern the challenge is trying to fix.
COOLDOWN_MINUTES = 45


def _minutes_between(earlier: Any, later: Any) -> Optional[float]:
    """Minutes from `earlier` to `later`, or None if either is unparseable."""
    a, b = g.parse_date(earlier), g.parse_date(later)
    if not a or not b:
        return None
    return (b - a).days * 24 * 60


def _cooldown_ok(merchant_id: str, now: str) -> bool:
    last = _global_dispatch.last_sent_to.get(merchant_id)
    if not last:
        return True
    minutes = _minutes_between(last, now)
    if minutes is None:
        # Unparseable clock — never block the run over a timestamp format.
        return True
    return minutes >= COOLDOWN_MINUTES


def _already_sent(suppression_key: str, now: str) -> bool:
    """True when this exact trigger has already been acted on.

    `suppression_key` is the judge's own dedup token, so honouring it is the
    contract. Without this, a trigger that stays in `available_triggers`
    across several ticks produces the same message repeatedly.
    """
    if not suppression_key:
        return False
    last = _global_dispatch.sent_suppressions.get(suppression_key)
    if not last:
        return False
    minutes = _minutes_between(last, now)
    if minutes is None:
        return True
    return minutes < 24 * 60  # once per simulated day


def conversation_id_for(trigger: dict, merchant: dict,
                        customer: Optional[dict]) -> str:
    """A decodable, resumable conversation id.

    The case studies note that `conv_001` is acceptable but a meaningful id
    helps; this encodes merchant, trigger family, and the customer when the
    send is on a customer's behalf.
    """
    raw = str(merchant.get("merchant_id", "unknown"))
    parts = raw.split("_")
    short_mid = "_".join(parts[:2]) if len(parts) >= 2 else raw
    family = re.sub(r"[^a-z0-9]+", "", str(trigger.get("kind", "trig")))[:16]
    if customer:
        cname = re.sub(r"[^a-z0-9]+", "",
                       str(g.dig(customer, "identity", "name", default="c") or "c"))[:12]
        return f"conv_{short_mid}_{family}_{cname}"
    return f"conv_{short_mid}_{family}"


def template_params(out: dict, merchant: dict, customer: Optional[dict]) -> list[str]:
    """Fill an approved-template shape for the first touch in a session.

    The judge never actually calls Meta, but it does check the action shape,
    so the params come from real context rather than placeholders.
    """
    name = (g.customer_first_name(customer) if customer
            else g.merchant_first_name(merchant))
    biz = str(g.dig(merchant, "identity", "name", default="") or "")
    return [name or biz or "there", out["body"], out.get("cta_text", "")]


def build_actions(available_triggers: list, now: str) -> list[dict[str, Any]]:
    """Compose the actions for this tick."""
    if not now:
        now = utc_now_iso()
    _global_dispatch.last_tick_at = now

    actions: list[dict[str, Any]] = []
    seen_merchants: set[str] = set()
    seen_conversations: set[str] = set()

    for trigger_id in available_triggers:
        if not isinstance(trigger_id, str):
            continue
        if len(actions) >= config.MAX_ACTIONS_PER_TICK:
            break

        trigger = contexts.get("trigger", trigger_id)
        if not trigger:
            # The judge may list a trigger before pushing it. Silence is right.
            continue
        if g.trigger_expired(trigger, now):
            continue

        merchant_id = str(trigger.get("merchant_id", "") or "")
        if not merchant_id or merchant_id in _global_dispatch.opted_out:
            continue
        # One action per merchant per tick.
        if merchant_id in seen_merchants:
            continue

        suppression_key = str(trigger.get("suppression_key", "") or "")
        if _already_sent(suppression_key, now):
            continue
        if not _cooldown_ok(merchant_id, now):
            continue

        merchant = contexts.get("merchant", merchant_id)
        if not merchant:
            continue

        category = contexts.get("category", str(merchant.get("category_slug", "")))
        if not category:
            continue

        customer = None
        customer_id = trigger.get("customer_id")
        if customer_id:
            customer = contexts.get("customer", str(customer_id))
            if not customer:
                # Customer-scoped trigger with no customer context yet. Skip
                # rather than send a half-informed message about a real person.
                continue

        # Reuse an open conversation for this trigger so multi-turn state and
        # anti-repetition carry across ticks.
        conv_id = conversation_id_for(trigger, merchant, customer)
        if conv_id in seen_conversations:
            continue
        conv = conversations.get(conv_id)
        if conv is not None and not conv.is_open():
            continue
        seen_conversations.add(conv_id)

        sent = set(conv.sent_bodies) if conv else set()
        out = compose(category, merchant, trigger, customer, now, sent)
        if out is None:
            continue

        conv = conversations.create(
            conversation_id=conv_id,
            merchant_id=merchant_id,
            customer_id=str(customer_id) if customer_id else None,
            trigger_id=trigger_id,
            send_as=out["send_as"],
        )
        conv.record_send(out["body"])

        _global_dispatch.sent_suppressions[suppression_key] = now
        _global_dispatch.last_sent_to[merchant_id] = now
        seen_merchants.add(merchant_id)

        kind = re.sub(r"[^a-z0-9]+", "_", str(trigger.get("kind", "generic"))).strip("_")
        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": str(customer_id) if customer_id else None,
            "send_as": out["send_as"],
            "trigger_id": trigger_id,
            "template_name": f"vera_{kind}_v1",
            "template_params": template_params(out, merchant, customer),
            "body": out["body"],
            "cta": out["cta"],
            "suppression_key": suppression_key,
            "rationale": out["rationale"],
        })

    return actions
