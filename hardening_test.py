#!/usr/bin/env python3
"""Hardening: determinism, latency, and malformed input.

The judge enforces three hard constraints that are easy to fail quietly:

  1. **Determinism** — "Your output should stay deterministic for the same
     input and simulator settings." Two identical runs must produce byte-identical
     action sets, including conversation_ids, template_params and rationales.
  2. **30s per request** — a slow /v1/tick or /v1/reply is an automatic zero,
     so measure p50 and worst case, not just the happy path.
  3. **Never 5xx** — the judge treats a 500 as a hard failure and the harness
     may abort the whole merchant.

    .venv/bin/python hardening_test.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

import server
from store import contexts, conversations, dispatch

DATASET = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")

GREEN, RED, DIM, CYAN, YELLOW, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[2m", "\033[96m", "\033[93m", "\033[1m", "\033[0m"
)

_passed: list[str] = []
_failed: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    (_passed if cond else _failed).append(name)
    mark = (GREEN + "[PASS]" + RESET) if cond else (RED + "[FAIL]" + RESET)
    print(f"  {mark} {name}" + (f" {DIM}{detail}{RESET}" if detail and not cond else ""))
    return cond


def section(t: str) -> None:
    print(f"\n{CYAN}{BOLD}--- {t} ---{RESET}")


def load(p: Path):
    return json.loads(p.read_text())


def fresh_client(seed_all: bool = True):
    """A brand-new app state, so runs cannot contaminate each other."""
    contexts.reset()
    conversations.reset()
    dispatch.reset()
    client = TestClient(server.app)
    if seed_all:
        for f in (DATASET / "categories").glob("*.json"):
            cat = load(f)
            client.post("/v1/context", json={"scope": "category",
                                              "context_id": cat["slug"],
                                              "version": 1, "payload": cat})
        for m in load(DATASET / "merchants_seed.json")["merchants"]:
            client.post("/v1/context", json={"scope": "merchant",
                                              "context_id": m["merchant_id"],
                                              "version": 1, "payload": m})
        for c in load(DATASET / "customers_seed.json")["customers"]:
            client.post("/v1/context", json={"scope": "customer",
                                              "context_id": c["customer_id"],
                                              "version": 1, "payload": c})
        for t in load(DATASET / "triggers_seed.json")["triggers"]:
            client.post("/v1/context", json={"scope": "trigger",
                                              "context_id": t["id"],
                                              "version": 1, "payload": t})
    return client


def normalise(actions: list[dict]) -> str:
    """Everything the judge can see, so any drift shows up."""
    return json.dumps(actions, sort_keys=True, ensure_ascii=False)


# --------------------------------------------------------------------------


def test_determinism() -> None:
    section("Determinism — identical input, identical output")
    tids = [t["id"] for t in load(DATASET / "triggers_seed.json")["triggers"]]
    runs = []
    for _ in range(3):
        c = fresh_client()
        a = c.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z",
                                     "available_triggers": tids}).json()["actions"]
        runs.append(normalise(a))
    check("3 independent runs are byte-identical", runs[0] == runs[1] == runs[2],
          "output drifted between runs")
    check("runs actually produced actions", len(json.loads(runs[0])) > 0)

    # Replay: once the 24h suppression window expires, sends legitimately resume
    # — that is the point of the window, not a determinism failure. What must
    # hold is that the resumed send is the *same* message, from the same
    # conversation, not a fresh random composition.
    c = fresh_client()
    a1 = c.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z",
                                  "available_triggers": tids}).json()["actions"]
    a2 = c.post("/v1/tick", json={"now": "2026-05-03T10:35:00Z",
                                  "available_triggers": tids}).json()["actions"]
    by_trigger_1 = {x["trigger_id"]: x for x in a1}
    by_trigger_2 = {x["trigger_id"]: x for x in a2}
    overlap = set(by_trigger_1) & set(by_trigger_2)
    check("suppression window expires, so sends resume", len(overlap) > 0,
          "nothing re-sent after a week — the 24h window never lifted")
    identical = all(by_trigger_1[t]["body"] == by_trigger_2[t]["body"]
                    for t in overlap)
    check("resumed sends are byte-identical to the originals", identical,
          "same trigger produced a different body later — non-deterministic")
    same_conv = all(by_trigger_1[t]["conversation_id"]
                    == by_trigger_2[t]["conversation_id"] for t in overlap)
    check("resumed sends reuse the same conversation_id", same_conv,
          "a new conversation was opened instead of resuming")

    section("Determinism — no wall-clock leakage")
    # A body built from datetime.now() would differ between runs on the same
    # context. Seed twice so both ticks see an identical, un-suppressed world,
    # then vary only `now` — any difference in the text is real leakage.
    #
    # Only time-INDEPENDENT trigger kinds are compared. A recall message that
    # says "8 weeks late" is *supposed* to change when the simulated clock
    # moves, and most seed triggers expire within days of the baseline so a
    # much later `now` legitimately drops them from the set entirely.
    TIME_INDEPENDENT = {"research_digest", "regulation_change", "competitor_opened"}
    seed_triggers = {t["id"]: t for t in load(DATASET / "triggers_seed.json")["triggers"]}
    probe = [tid for tid, t in seed_triggers.items() if t["kind"] in TIME_INDEPENDENT]
    check("found time-independent triggers to probe", len(probe) >= 2,
          f"only {len(probe)}")

    c1 = fresh_client()
    a = c1.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z",
                                  "available_triggers": probe}).json()["actions"]
    c2 = fresh_client()
    b = c2.post("/v1/tick", json={"now": "2026-04-26T18:35:00Z",
                                  "available_triggers": probe}).json()["actions"]
    bodies_a = {x["trigger_id"]: x["body"] for x in a}
    bodies_b = {x["trigger_id"]: x["body"] for x in b}
    shared = set(bodies_a) & set(bodies_b)
    check("both ticks produced comparable output", len(shared) > 0,
          f"no overlapping triggers ({len(bodies_a)} vs {len(bodies_b)})")
    same = [t for t in shared if bodies_a[t] == bodies_b[t]]
    check("composed text does not embed the real current date",
          len(same) == len(shared),
          f"{len(shared) - len(same)}/{len(shared)} bodies changed with the clock")
    joined = " ".join(bodies_a.values()).lower()
    check("no relative-time wording that presumes a real 'now'",
          not any(w in joined for w in ("just now", "as of right now",
                                        "in a moment", "at this moment")),
          "message implies wall-clock time")


def test_latency() -> None:
    section("Latency — budget is 30s per request")
    tids = [t["id"] for t in load(DATASET / "triggers_seed.json")["triggers"]]
    c = fresh_client()
    samples = []
    for _ in range(12):
        t0 = time.perf_counter()
        c.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z",
                                 "available_triggers": tids})
        samples.append((time.perf_counter() - t0) * 1000)
    p50, worst = statistics.median(samples), max(samples)
    print(f"    {DIM}/v1/tick  p50 {p50:.0f}ms   worst {worst:.0f}ms{RESET}")
    check("tick p50 well under budget", p50 < 500, f"{p50:.0f}ms")
    check("tick worst case under 30s", worst < 30_000, f"{worst:.0f}ms")

    r_samples = []
    for i in range(12):
        t0 = time.perf_counter()
        c.post("/v1/reply", json={
            "conversation_id": f"conv_perf_{i}", "merchant_id":
            "m_001_drmeera_dentist_delhi", "customer_id": None,
            "from_role": "merchant", "message": "ok do it",
            "received_at": "2026-04-26T10:45:00Z", "turn_number": 2})
        r_samples.append((time.perf_counter() - t0) * 1000)
    p50r, worstr = statistics.median(r_samples), max(r_samples)
    print(f"    {DIM}/v1/reply p50 {p50r:.0f}ms   worst {worstr:.0f}ms{RESET}")
    check("reply p50 well under budget", p50r < 500, f"{p50r:.0f}ms")
    check("reply worst case under 30s", worstr < 30_000, f"{worstr:.0f}ms")

    section("Scale — full 100-trigger dataset in one tick")
    # 75 of the 100 triggers are generated placeholders, which is the worst case.
    c2 = fresh_client()
    gen = []
    for i in range(75):
        tid = f"trg_gen_{i:03d}"
        c2.post("/v1/context", json={"scope": "trigger", "context_id": tid,
                                     "version": 1, "payload": {
            "id": tid, "scope": "merchant", "kind": "research_digest",
            "source": "generated", "merchant_id": f"m_{(i % 50) + 1:03d}_x",
            "customer_id": None, "payload": {"placeholder": True},
            "urgency": 2, "suppression_key": f"gen:{i}",
            "expires_at": "2026-12-01T00:00:00Z"}})
        gen.append(tid)
    t0 = time.perf_counter()
    r = c2.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z",
                                  "available_triggers": gen})
    ms = (time.perf_counter() - t0) * 1000
    acts = r.json().get("actions", [])
    print(f"    {DIM}75 placeholder triggers -> {len(acts)} actions in {ms:.0f}ms{RESET}")
    check("100-trigger tick returns 200", r.status_code == 200, str(r.status_code))
    check("100-trigger tick under 30s", ms < 30_000, f"{ms:.0f}ms")
    check("never exceeds the 20-action cap", len(acts) <= 20, f"{len(acts)}")
    check("unknown merchants produce no actions", len(acts) == 0,
          f"{len(acts)} actions from merchants that do not exist")


def test_malformed() -> None:
    section("Malformed input — the judge must never see a 5xx")
    c = fresh_client(seed_all=False)
    hostile = [
        ("empty body", {}),
        ("null body", None),
        ("wrong types everywhere", {"now": 12345, "available_triggers": "not-a-list"}),
        ("trigger list of ints", {"now": "2026-04-26T10:35:00Z",
                                  "available_triggers": [1, 2, 3]}),
        ("trigger list of dicts", {"now": "2026-04-26T10:35:00Z",
                                   "available_triggers": [{"a": 1}]}),
        ("nested nulls", {"now": None, "available_triggers": [None, None]}),
        ("huge trigger list", {"now": "2026-04-26T10:35:00Z",
                               "available_triggers": ["t"] * 5000}),
        ("bad date format", {"now": "not-a-date", "available_triggers": []}),
        ("impossible date", {"now": "99999-99-99T99:99:99Z",
                             "available_triggers": []}),
        ("missing available_triggers", {"now": "2026-04-26T10:35:00Z"}),
        ("unicode in ids", {"now": "2026-04-26T10:35:00Z",
                            "available_triggers": ["\U0001f600", "Ω", "../../etc/passwd"]}),
    ]
    for name, payload in hostile:
        try:
            r = c.post("/v1/tick", json=payload)
            ok = r.status_code < 500
        except Exception as e:  # noqa: BLE001
            ok, r = False, e
        check(f"tick: {name} -> not 5xx", ok, str(r)[:90])

    hostile_reply = [
        ("empty body", {}),
        ("no message", {"conversation_id": "c1"}),
        ("message is null", {"conversation_id": "c1", "message": None}),
        ("message is an int", {"conversation_id": "c1", "message": 42}),
        ("message is a list", {"conversation_id": "c1", "message": ["a", "b"]}),
        ("turn_number is a string", {"conversation_id": "c1", "message": "hi",
                                     "turn_number": "two"}),
        ("unknown from_role", {"conversation_id": "c1", "message": "hi",
                               "from_role": "martian"}),
        ("1MB message", {"conversation_id": "c1", "message": "x" * 1_000_000}),
        ("emoji only", {"conversation_id": "c1", "message": "\U0001f600\U0001f600"}),
        ("script injection", {"conversation_id": "c1",
                              "message": "<script>alert(1)</script>"}),
        ("sql-ish", {"conversation_id": "c1", "message": "'; DROP TABLE--"}),
        ("path traversal in conv id", {"conversation_id": "../../etc/passwd",
                                       "message": "hi"}),
        ("newlines in message", {"conversation_id": "c1", "message": "a\nb\nc\nd"}),
    ]
    for name, payload in hostile_reply:
        try:
            r = c.post("/v1/reply", json=payload)
            ok = r.status_code < 500
        except Exception as e:  # noqa: BLE001
            ok, r = False, e
        check(f"reply: {name} -> not 5xx", ok, str(r)[:90])

    section("Malformed context pushes")
    for name, payload in [
        ("payload is a list", {"scope": "category", "context_id": "x",
                               "version": 1, "payload": [1, 2, 3]}),
        ("payload is a string", {"scope": "category", "context_id": "x",
                                 "version": 1, "payload": "hello"}),
        ("version is negative", {"scope": "category", "context_id": "x",
                                 "version": -5, "payload": {}}),
        ("version is a string", {"scope": "category", "context_id": "x",
                                 "version": "one", "payload": {}}),
        ("deeply nested payload", {"scope": "merchant", "context_id": "x",
                                   "version": 1,
                                   "payload": {"a": {"b": {"c": {"d": {"e": 1}}}}}}),
    ]:
        try:
            r = c.post("/v1/context", json=payload)
            ok = r.status_code < 500
        except Exception as e:  # noqa: BLE001
            ok, r = False, e
        check(f"context: {name} -> not 5xx", ok, str(r)[:90])

    section("Contradictory / nonsense context")
    c2 = fresh_client(seed_all=False)
    cases = [
        ("digest item with no title", {"slug": "x", "digest": [{"id": "d1"}]}),
        ("trigger with expires_at in the past", {
            "id": "t1", "scope": "merchant", "kind": "research_digest",
            "merchant_id": "m1", "payload": {"category": "x"},
            "expires_at": "2020-01-01T00:00:00Z"}),
        ("merchant with null performance", {
            "merchant_id": "m1", "identity": {"name": "X"},
            "performance": None}),
        ("customer with no identity", {
            "customer_id": "c1", "identity": None, "state": "lapsed"}),
        ("offer with negative price", {
            "merchant_id": "m1", "offers": [{"id": "o1", "title": "Off",
                                              "price": -500}]}),
        ("payload claiming placeholder", {
            "id": "t2", "scope": "merchant", "kind": "recall_due",
            "merchant_id": "m1", "customer_id": "c1",
            "payload": {"placeholder": True}}),
    ]
    for name, payload in cases:
        scope = ("trigger" if "id" in payload and "scope" in payload
                 else "customer" if "customer_id" in payload
                 else "merchant" if "merchant_id" in payload
                 else "category")
        cid = str(payload.get("id") or payload.get("merchant_id")
                  or payload.get("customer_id") or "x")
        try:
            push = c2.post("/v1/context", json={"scope": scope, "context_id": cid,
                                                "version": 1, "payload": payload})
            tid = cid if scope == "trigger" else None
            tick = c2.post("/v1/tick", json={
                "now": "2026-04-26T10:35:00Z",
                "available_triggers": [tid] if tid else []})
            ok = push.status_code < 500 and tick.status_code < 500
            detail = f"push {push.status_code} tick {tick.status_code}"
        except Exception as e:  # noqa: BLE001
            ok, detail = False, str(e)[:90]
        check(f"{name} -> survives push + tick", ok, detail)


def main() -> int:
    print(f"{BOLD}{'=' * 62}{RESET}\n  Vera bot — hardening\n{BOLD}{'=' * 62}{RESET}")
    test_determinism()
    test_latency()
    test_malformed()
    total = len(_passed) + len(_failed)
    print(f"\n{BOLD}{'=' * 62}{RESET}")
    if _failed:
        print(f"  {RED}{len(_failed)}/{total} FAILED{RESET}")
        for n in _failed:
            print(f"    {RED}x{RESET} {n}")
        return 1
    print(f"  {GREEN}ALL {total} HARDENING CHECKS PASSED{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
