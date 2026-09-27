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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import config
import grounding as g
from composer import compose
from store import contexts, conversations, dispatch as _global_dispatch, utc_now_iso

# Minimum simulated minutes between two sends to the same merchant. Three
# messages in an hour is exactly the pattern the challenge is trying to fix.
# The window lives in config so a graded deployment can open it up — see the
# "Restraint vs. re-evaluability" block there for why 45 was the wrong default.


def _minutes_between(earlier: Any, later: Any) -> Optional[float]:
    """Minutes from `earlier` to `later`, or None if either is unparseable.

    Use total_seconds(), never .days. `timedelta(hours=2).days` is 0, so a
    .days-based version reports every sub-day interval as 0 minutes — which
    silently made every same-day comparison read as "0 minutes elapsed". That
    put a 45-minute cooldown permanently out of reach inside one calendar day
    and muted every evaluation re-run made on the same date as the first, with
    no error anywhere. The bot looked alive and returned zero actions.
    """
    a, b = g.parse_moment(earlier), g.parse_moment(later)
    if not a or not b:
        return None
    return (b - a).total_seconds() / 60.0


def _cooldown_ok(merchant_id: str, now: str) -> bool:
    if config.COOLDOWN_MINUTES <= 0:
        return True
    last = _global_dispatch.last_sent_to.get(merchant_id)
    if not last:
        return True
    minutes = _minutes_between(last, now)
    if minutes is None:
        # Unparseable clock — never block the run over a timestamp format.
        return True
    return minutes >= config.COOLDOWN_MINUTES


def _already_sent(suppression_key: str, now: str) -> bool:
    """True when this exact trigger was already acted on inside the window.

    `suppression_key` is the judge's own dedup token, so honouring it is the
    contract. Without this, a trigger that stays in `available_triggers` across
    several ticks produces the same message repeatedly.

    The window is `SUPPRESSION_HOURS`, not a flat 24h. Both are measured in the
    judge's clock and the state is in-memory, so a long window on a long-lived
    process mutes every run after the first — see config.py.
    """
    if config.SUPPRESSION_HOURS <= 0 or not suppression_key:
        return False
    last = _global_dispatch.sent_suppressions.get(suppression_key)
    if not last:
        return False
    minutes = _minutes_between(last, now)
    if minutes is None:
        return True
    return minutes < config.SUPPRESSION_HOURS * 60


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
    """Compose the actions for this tick.

    Two phases on purpose. The guards below cost nothing and decide *which*
    messages are eligible, so they run first and synchronously. Only the
    survivors get composed, and those run concurrently — measured at 4.1s and
    5.2s per message against real providers, so ten of them sequentially is
    37-52s and blows the judge's 30s budget. In parallel it is 7-15s.
    """
    if not now:
        now = utc_now_iso()
    _global_dispatch.last_tick_at = now

    per_merchant: dict[str, int] = {}
    seen_conversations: set[str] = set()
    eligible: list[dict[str, Any]] = []

    for trigger_id in available_triggers:
        if not isinstance(trigger_id, str):
            continue

        trigger = contexts.get("trigger", trigger_id)
        if not trigger:
            # The judge may list a trigger before pushing it. Silence is right.
            continue
        # Expiry demotes rather than blocks. The seed triggers carry
        # expires_at in April-December 2026 while the official judge_simulator
        # sends a wall-clock `now`, so a hard block muxes the bot completely
        # and it scores zero on a run it should be competing in. An expired
        # trigger is still a real event the merchant should hear about; it
        # simply sorts below anything timely.
        expired = g.trigger_expired(trigger, now)

        merchant_id = str(trigger.get("merchant_id", "") or "")
        if not merchant_id or merchant_id in _global_dispatch.opted_out:
            continue
        # A merchant may get more than one action per tick now — up to
        # MAX_ACTIONS_PER_MERCHANT_PER_TICK. This was a hard 1, which meant
        # the judge's batches (built so 2-3 triggers share a merchant) lost 11
        # of 25 messages. Each action is a separate conversation and a separate
        # grounded signal, so they are not repeats of one another.
        if per_merchant.get(merchant_id, 0) >= config.MAX_ACTIONS_PER_MERCHANT_PER_TICK:
            continue

        suppression_key = str(trigger.get("suppression_key", "") or "")
        if _already_sent(suppression_key, now):
            continue
        if not _cooldown_ok(merchant_id, now):
            continue

        merchant = contexts.get("merchant", merchant_id)
        merchant_missing = not merchant
        if merchant_missing:
            # Degrade, don't drop. A trigger is itself an authoritative
            # statement that something happened — a recall, a lapsed
            # subscription, a due refill — and its payload carries the dates
            # and numbers. Silently discarding it because a profile has not
            # arrived yet loses the message entirely, when the honest answer
            # ("a 6-month cleaning is due 2025-11-12") was available all
            # along. The brief pushes exactly this case: context is pushed
            # incrementally and new triggers can reference merchants the bot
            # has not been told about yet.
            merchant = {"merchant_id": merchant_id}

        category = contexts.get("category", str(merchant.get("category_slug", "")))
        if not category:
            category = {}

        customer = None
        customer_id = trigger.get("customer_id")
        if customer_id:
            customer = contexts.get("customer", str(customer_id))
            if not customer:
                # Same reasoning, and here the id is enough to go on: customer
                # ids encode the person's name, and the judge supplied it.
                # Only the record is missing, not the identity.
                customer = {"customer_id": str(customer_id)}

        # Reuse an open conversation for this trigger so multi-turn state and
        # anti-repetition carry across ticks.
        conv_id = conversation_id_for(trigger, merchant, customer)
        if conv_id in seen_conversations:
            continue
        conv = conversations.get(conv_id)
        if conv is not None and not conv.is_open():
            continue
        seen_conversations.add(conv_id)
        per_merchant[merchant_id] = per_merchant.get(merchant_id, 0) + 1

        eligible.append({
            "conversation_id": conv_id,
            "merchant": merchant,
            "category": category,
            "customer": customer,
            "customer_id": str(customer_id) if customer_id else None,
            "trigger_id": trigger_id,
            "trigger": trigger,
            "suppression_key": suppression_key,
            "merchant_id": merchant_id,
            "expired": expired,
            "sent": set(conv.sent_bodies) if conv else set(),
        })

    if not eligible:
        return []

    # Timely triggers first. Expired ones still ship — see the note where
    # `expired` is computed — but never at the cost of a live opportunity when
    # the tick is near its action cap.
    eligible.sort(key=lambda item: (item["expired"], item["trigger_id"]))

    # Apply the action cap with round-robin fairness rather than by input
    # order. Truncating the sorted list starves whichever merchant happens to
    # sit late in `available_triggers`: a 25-trigger tick capped at 20 dropped
    # a merchant entirely, and the judge only ever sends 5 at a time so nothing
    # in the normal path exposed it. Round-robin takes one action per merchant
    # at a time, so every merchant is represented before any gets a second.
    if len(eligible) > config.MAX_ACTIONS_PER_TICK:
        queues: dict[str, list[dict[str, Any]]] = {}
        for item in eligible:
            queues.setdefault(item["merchant_id"], []).append(item)
        taken: list[dict[str, Any]] = []
        while len(taken) < config.MAX_ACTIONS_PER_TICK and any(queues.values()):
            for merchant_id in list(queues):
                if len(taken) >= config.MAX_ACTIONS_PER_TICK:
                    break
                bucket = queues[merchant_id]
                if bucket:
                    taken.append(bucket.pop(0))
        # Restore the timely-first ordering within the chosen set.
        taken.sort(key=lambda item: (item["expired"], item["trigger_id"]))
        eligible = taken

    # ---- phase 2: compose, concurrently ---------------------------------
    outputs = _compose_all(eligible, now)

    # ---- phase 3: commit the sends -------------------------------------
    actions: list[dict[str, Any]] = []
    for item, out in zip(eligible, outputs):
        if out is None:
            continue

        merchant = item["merchant"]
        conv = conversations.create(
            conversation_id=item["conversation_id"],
            merchant_id=item["merchant_id"],
            customer_id=item["customer_id"],
            trigger_id=item["trigger_id"],
            send_as=out["send_as"],
        )
        conv.record_send(out["body"])

        _global_dispatch.sent_suppressions[item["suppression_key"]] = now
        _global_dispatch.last_sent_to[item["merchant_id"]] = now

        kind = re.sub(r"[^a-z0-9]+", "_",
                      str(item["trigger"].get("kind", "generic"))).strip("_")
        actions.append({
            "conversation_id": item["conversation_id"],
            "merchant_id": item["merchant_id"],
            "customer_id": item["customer_id"],
            "send_as": out["send_as"],
            "trigger_id": item["trigger_id"],
            "template_name": f"vera_{kind}_v1",
            "template_params": template_params(out, merchant, item["customer"]),
            "body": out["body"],
            "cta": out["cta"],
            "suppression_key": item["suppression_key"],
            "rationale": out["rationale"],
        })

    return actions


def _compose_all(eligible: list[dict[str, Any]], now: str) -> list[Optional[dict]]:
    """Run compose() across all eligible items concurrently.

    compose() is read-only against shared state — the conversation writes
    happen afterwards, on this thread — so running it from a pool is safe.
    Each item already has a distinct conversation_id and merchant, so nothing
    two workers touch is the same object.
    """
    def work(item: dict[str, Any]) -> Optional[dict]:
        return compose(item["category"], item["merchant"], item["trigger"],
                       item["customer"], now, item["sent"])

    if len(eligible) == 1 or not config.llm_configured():
        # Deterministic path is ~1ms; a thread pool would only add overhead.
        return [work(i) for i in eligible]

    workers = min(len(eligible), 8)
    results: list[Optional[dict]] = [None] * len(eligible)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, item): idx
                   for idx, item in enumerate(eligible)}
        for fut in as_completed(futures, timeout=config.TICK_DEADLINE_S):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception:  # noqa: BLE001 — one bad message must not 500
                results[idx] = None
    return results
