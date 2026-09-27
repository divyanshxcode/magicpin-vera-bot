#!/usr/bin/env python3
"""Replay-test the three scenarios magicpin runs on the top 10 bots.

From challenge-testing-brief.md §4, Phase 4:

  1. **Auto-reply hell** — the same canned text 4 times. Must detect and exit.
  2. **Intent transition** — after qualification, "ok let's do it". Must switch
     to action, not ask another question.
  3. **Hostile / off-topic** — abuse, then an unrelated question. Must stay
     on mission politely.

These are scored on conversation flow only, independent of message quality, so
they are worth testing directly rather than inferring from the smoke test.

    .venv/bin/python replay_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

import server
from store import conversations, dispatch

DATASET = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")

GREEN, RED, YELLOW, DIM, CYAN, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[96m", "\033[1m", "\033[0m"
)

_passed: list[str] = []
_failed: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    (_passed if cond else _failed).append(name)
    print(f"  {(GREEN + '[PASS]' + RESET) if cond else (RED + '[FAIL]' + RESET)} {name}"
          + (f" {DIM}{detail}{RESET}" if detail and not cond else ""))
    return cond


def section(t: str) -> None:
    print(f"\n{CYAN}{BOLD}--- {t} ---{RESET}")


def load(p: Path):
    return json.loads(p.read_text())


def seed(client: TestClient) -> str:
    """Load the full seed dataset; return a merchant id to talk to."""
    for f in (DATASET / "categories").glob("*.json"):
        cat = load(f)
        client.post("/v1/context", json={"scope": "category", "context_id": cat["slug"],
                                          "version": 1, "payload": cat})
    merchants = load(DATASET / "merchants_seed.json")["merchants"]
    for m in merchants:
        client.post("/v1/context", json={"scope": "merchant",
                                          "context_id": m["merchant_id"],
                                          "version": 1, "payload": m})
    for c in load(DATASET / "customers_seed.json")["customers"]:
        client.post("/v1/context", json={"scope": "customer",
                                          "context_id": c["customer_id"],
                                          "version": 1, "payload": c})
    return merchants[0]["merchant_id"]


def reply(client: TestClient, conv: str, mid: str, msg: str, turn: int) -> dict:
    r = client.post("/v1/reply", json={
        "conversation_id": conv, "merchant_id": mid, "customer_id": None,
        "from_role": "merchant", "message": msg,
        "received_at": "2026-04-26T10:45:00Z", "turn_number": turn})
    return r.json()


# --------------------------------------------------------------------------


def test_auto_reply_hell(client: TestClient, mid: str) -> None:
    section("Replay 1 — auto-reply hell (4x identical canned text)")
    conv = "conv_replay_autoreply"
    canned = ("Thank you for contacting Dr. Meera's Dental Clinic! "
              "Our team will respond shortly.")
    actions = []
    for turn in range(2, 6):
        d = reply(client, conv, mid, canned, turn)
        action = d.get("action")
        actions.append(action)
        body = (d.get("body") or "")[:64]
        print(f"    turn {turn}: {YELLOW}{action}{RESET}"
              + (f"  {DIM}{body}{RESET}" if body else
                 (f"  {DIM}wait {d.get('wait_seconds')}s{RESET}" if action == "wait" else "")))

    check("responds to every turn (never silent)", len(actions) == 4)
    check("ends the conversation by turn 4", actions[-1] == "end",
          f"final action was {actions[-1]}")
    check("does not send 4 real messages", actions.count("send") <= 1,
          f"sent {actions.count('send')}x — Vera's original failure mode")
    check("backs off with wait before ending", "wait" in actions,
          "went straight to end without trying the owner")
    check("merchant not permanently opted out by an auto-reply",
          mid not in dispatch.opted_out, "auto-reply must not opt the merchant out")


def test_intent_transition(client: TestClient, mid: str) -> None:
    section("Replay 2 — intent transition (commit must switch to action)")
    conv = "conv_replay_intent"
    reply(client, conv, mid, "What can you do for my listing?", 2)
    d = reply(client, conv, mid, "Ok lets do it. Whats next?", 3)
    body = (d.get("body") or "")
    print(f"    action: {YELLOW}{d.get('action')}{RESET}")
    print(f"    body  : {DIM}{body}{RESET}")

    low = body.lower()
    check("acts rather than stalling", d.get("action") == "send", str(d.get("action")))
    check("does NOT keep qualifying after a commit",
          not any(w in low for w in ("would you", "do you think", "can you tell",
                                     "what if", "how about", "just to plan")),
          "still asking qualifying questions after 'lets do it' — the named failure")
    check("produces something concrete",
          any(w in low for w in ("draft", "sending", "here", "confirm", "go ahead", "now")),
          "no deliverable offered")
    check("gives a low-friction next action",
          any(w in low for w in ("confirm", "reply", "tell me")), "no CTA")


def test_hostile_offtopic(client: TestClient, mid: str) -> None:
    section("Replay 3 — hostile, then off-topic")
    conv = "conv_replay_hostile"
    d1 = reply(client, conv, mid,
               "Why are you bothering me. This is useless. Stop sending these.", 2)
    print(f"    hostile  -> {YELLOW}{d1.get('action')}{RESET}  "
          f"{DIM}{(d1.get('body') or '')[:64]}{RESET}")
    check("ends or apologises on hostility",
          d1.get("action") == "end" or "sorry" in (d1.get("body") or "").lower(),
          str(d1.get("action")))

    conv2 = "conv_replay_offtopic"
    d2 = reply(client, conv2, mid,
               "Btw can you also help me with my GST filing this month?", 2)
    body2 = (d2.get("body") or "")
    print(f"    offtopic -> {YELLOW}{d2.get('action')}{RESET}  {DIM}{body2[:64]}{RESET}")
    check("answers the off-topic ask honestly", d2.get("action") == "send", str(d2.get("action")))
    check("declines rather than pretending to help",
          any(w in body2.lower() for w in ("ca", "accountant", "not something i can",
                                           "outside", "can't", "cannot")),
          f"did not decline: {body2[:80]}")
    check("stays in scope (no GST advice given)",
          not any(w in body2.lower() for w in ("file your gst", "i filed", "i can file",
                                               "submit your")),
          "started doing the out-of-scope work")


def test_optout_persists(client: TestClient, mid: str) -> None:
    section("Extra — explicit opt-out is permanent")
    conv = "conv_replay_optout"
    d = reply(client, conv, mid, "Stop messaging me.", 2)
    check("ends on opt-out", d.get("action") == "end", str(d.get("action")))
    check("merchant added to opt-out list", mid in dispatch.opted_out)

    d2 = reply(client, conv, mid, "hello?", 3)
    check("stays ended on the same conversation", d2.get("action") == "end",
          str(d2.get("action")))


def main() -> int:
    print(f"{BOLD}{'=' * 62}{RESET}\n  Vera bot — replay scenarios\n{BOLD}{'=' * 62}{RESET}")
    client = TestClient(server.app)
    mid = seed(client)
    print(f"{DIM}seeded; merchant under test: {mid}{RESET}")

    test_auto_reply_hell(client, mid)
    test_intent_transition(client, mid)
    test_hostile_offtopic(client, mid)
    test_optout_persists(client, mid)

    total = len(_passed) + len(_failed)
    print(f"\n{BOLD}{'=' * 62}{RESET}")
    if _failed:
        print(f"  {RED}{len(_failed)}/{total} FAILED{RESET}")
        for n in _failed:
            print(f"    {RED}x{RESET} {n}")
        return 1
    print(f"  {GREEN}ALL {total} REPLAY CHECKS PASSED{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
