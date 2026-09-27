#!/usr/bin/env python3
"""A second evaluation run must not be muted by the first.

Regression test for the deployed-service failure. `suppression_key` is the
judge's own dedup token and the brief requires honouring it, but the window it
was honoured over was 24 simulated hours, measured in the clock the judge
supplies and backed by in-memory state. On Render — which stays warm between
attempts instead of recycling like a cold start does — that meant run one
scored and every run after it returned {"actions": []} for the rest of the day,
silently, with /v1/healthz still reporting all context loaded.

The fix separates the two things that were conflated:
  - dedup, which the brief asks for, defaults to a short window
  - daily restraint, which nobody asked for, is opt-in and disabled in render.yaml

These checks pin the behaviour that actually matters, in both configurations.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

import config
import server
from store import contexts, conversations, dispatch

DATASET = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")

GREEN, RED, DIM, BOLD, RESET = "\033[92m", "\033[91m", "\033[2m", "\033[1m", "\033[0m"

passed = 0
total = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, total
    total += 1
    passed += ok
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {name}{DIM}  {detail}{RESET}")


def fresh() -> TestClient:
    for s in (contexts, conversations, dispatch):
        s.reset()
    return TestClient(server.app)


def seed(c: TestClient, n: int = 5) -> list[str]:
    for f in (DATASET / "categories").glob("*.json"):
        x = json.loads(f.read_text())
        c.post("/v1/context", json={"scope": "category", "context_id": x["slug"],
                                    "version": 1, "payload": x})
    merchants = json.loads((DATASET / "merchants_seed.json").read_text())["merchants"]
    for m in merchants[:n]:
        c.post("/v1/context", json={"scope": "merchant", "context_id": m["merchant_id"],
                                    "version": 1, "payload": m})
    triggers = json.loads((DATASET / "triggers_seed.json").read_text())["triggers"]
    picked, seen = [], set()
    for t in triggers:
        if t["merchant_id"] in seen:
            continue
        c.post("/v1/context", json={"scope": "trigger", "context_id": t["id"],
                                    "version": 1, "payload": t})
        picked.append(t["id"])
        seen.add(t["merchant_id"])
        if len(picked) == n:
            break
    return picked


def tick(c: TestClient, ids: list[str], now: str) -> list[dict]:
    return c.post("/v1/tick", json={"now": now,
                                    "available_triggers": ids}).json().get("actions", [])


def with_config(sup_h: float, cool_m: int):
    """Swap the two windows for the duration of a block."""
    old = (config.SUPPRESSION_HOURS, config.COOLDOWN_MINUTES)
    config.SUPPRESSION_HOURS, config.COOLDOWN_MINUTES = sup_h, cool_m
    return old


def main() -> int:
    print(f"{BOLD}A second evaluation run must not be muted by the first{RESET}\n")

    # ---------------------------------------------------------------------
    print(f"{DIM}  as deployed: both windows default to disabled{RESET}")
    # No with_config() here on purpose. This asserts the SHIPPED default, which
    # is what runs on Render whether or not render.yaml's envVars arrived — and
    # they do not arrive for a manually-created service.
    check("SUPPRESSION_HOURS defaults to 0", config.SUPPRESSION_HOURS == 0.0,
          f"{config.SUPPRESSION_HOURS}")
    check("COOLDOWN_MINUTES defaults to 0", config.COOLDOWN_MINUTES == 0,
          f"{config.COOLDOWN_MINUTES}")
    check("defaults are not the old 24h/45m",
          not (config.SUPPRESSION_HOURS == 24.0 or config.COOLDOWN_MINUTES == 45),
          f"{config.SUPPRESSION_HOURS}h / {config.COOLDOWN_MINUTES}min")

    c = fresh()
    ids = seed(c)
    r1 = tick(c, ids, "2026-04-26T10:35:00Z")
    check("first run produces actions", len(r1) > 0, f"{len(r1)} actions")

    # The judge supplies wall-clock `now`, so a re-run minutes later looks
    # like this: same triggers, a later clock, no teardown and no restart.
    r2 = tick(c, ids, "2026-04-26T10:52:00Z")
    check("re-run 17 min later still produces actions", len(r2) > 0, f"{len(r2)} actions")
    check("re-run returns the same messages",
          {a["trigger_id"] for a in r1} == {a["trigger_id"] for a in r2},
          f"{len(r1)} vs {len(r2)}")
    check("re-run at the SAME instant is not suppressed either",
          len(tick(c, ids, "2026-04-26T10:52:00Z")) > 0, "identical clock re-tick")
    r3 = tick(c, ids, "2026-04-26T14:35:00Z")
    check("and again 4 hours later", len(r3) > 0, f"{len(r3)} actions")

    # ---------------------------------------------------------------------
    print(f"\n{DIM}  with windows enabled, dedup still does its job{RESET}")
    c = fresh()
    ids = seed(c)
    old = with_config(1.0, 15)
    try:
        a1 = tick(c, ids, "2026-04-26T10:35:00Z")
        check("first run produces actions", len(a1) > 0, f"{len(a1)} actions")

        # Same clock, immediate re-tick: this is the dedup the brief requires.
        a2 = tick(c, ids, "2026-04-26T10:35:00Z")
        check("immediate re-tick at the same clock is deduplicated",
              len(a2) == 0, f"{len(a2)} actions")

        # 20 minutes on: past the 15-minute cooldown, still inside the 1h
        # suppression window, so the per-trigger dedup is what's doing the work.
        a3 = tick(c, ids, "2026-04-26T10:55:00Z")
        check("20 min later, trigger still suppressed (dedup intact)",
              len(a3) == 0, f"{len(a3)} actions")

        # Past the 1h suppression window: sends legitimately resume.
        a4 = tick(c, ids, "2026-04-26T12:35:00Z")
        check("past SUPPRESSION_HOURS, sends resume", len(a4) > 0, f"{len(a4)} actions")
        check("resumed sends are the same triggers, not a fresh set",
              {x["trigger_id"] for x in a4} == {x["trigger_id"] for x in a1},
              "resumed set differs from the original")
    finally:
        config.SUPPRESSION_HOURS, config.COOLDOWN_MINUTES = old

    # ---------------------------------------------------------------------
    print(f"\n{DIM}  the elapsed-time helper counts minutes, not whole days{RESET}")
    # timedelta(hours=2).days == 0, so a .days-based helper reports every
    # sub-day interval as 0. That is what muted every same-day re-run.
    import tick as _tick
    for lo, hi, want in [("10:35", "10:35", 0.0), ("10:35", "10:55", 20.0),
                         ("10:35", "12:35", 120.0), ("10:35", "23:55", 800.0)]:
        got = _tick._minutes_between(f"2026-04-26T{lo}:00Z", f"2026-04-26T{hi}:00Z")
        check(f"{lo} -> {hi} is {want:g}m", got is not None and abs(got - want) < 1e-6,
              f"got {got}")
    got = _tick._minutes_between("2026-04-26T10:35:00Z", "2026-04-27T10:35:00Z")
    check("across a day boundary it is 1440m", got is not None and abs(got - 1440) < 1e-6,
          f"got {got}")

    # ---------------------------------------------------------------------
    print(f"\n{DIM}  the old 24h behaviour really did mute a re-run{RESET}")
    c = fresh()
    ids = seed(c)
    old = with_config(24.0, 45)
    try:
        tick(c, ids, "2026-04-26T10:35:00Z")
        muted = tick(c, ids, "2026-04-26T10:52:00Z")
        check("24h/45m reproduces the silent mute", len(muted) == 0,
              f"{len(muted)} actions — this is the bug")
    finally:
        config.SUPPRESSION_HOURS, config.COOLDOWN_MINUTES = old

    print(f"\n  {GREEN if passed == total else RED}{passed}/{total} checks passed{RESET}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
