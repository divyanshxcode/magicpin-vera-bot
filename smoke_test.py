#!/usr/bin/env python3
"""End-to-end smoke test for the Vera bot's HTTP contract.

Runs the real ASGI app in-process (no server needed) and asserts the exact
response shapes the judge harness depends on. Grows with each phase; run it
after every change:

    .venv/bin/python smoke_test.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from fastapi.testclient import TestClient

logging.getLogger("httpx").setLevel(logging.WARNING)

import server
from store import contexts, conversations, dispatch

DATASET = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")

GREEN, RED, YELLOW, DIM, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"

_passed: list[str] = []
_failed: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    if cond:
        _passed.append(name)
        print(f"  {GREEN}[PASS]{RESET} {name}")
    else:
        _failed.append(name)
        print(f"  {RED}[FAIL]{RESET} {name} {DIM}{detail}{RESET}")
    return cond


def section(title: str) -> None:
    print(f"\n{YELLOW}--- {title} ---{RESET}")


def load(path: Path):
    return json.loads(path.read_text())


# --------------------------------------------------------------------------


def test_healthz_metadata(client: TestClient) -> None:
    section("healthz + metadata")
    r = client.get("/v1/healthz")
    check("healthz returns 200", r.status_code == 200, f"got {r.status_code}")
    d = r.json()
    check("healthz has status/uptime/contexts_loaded",
          all(k in d for k in ("status", "uptime_seconds", "contexts_loaded")), str(d))
    check("status == ok", d.get("status") == "ok", str(d.get("status")))
    check("contexts_loaded has all 4 scope keys",
          set(d.get("contexts_loaded", {})) == {"category", "merchant", "customer", "trigger"},
          str(d.get("contexts_loaded")))
    check("fresh bot has zero contexts loaded",
          all(v == 0 for v in d["contexts_loaded"].values()), str(d["contexts_loaded"]))

    r = client.get("/v1/metadata")
    check("metadata returns 200", r.status_code == 200, f"got {r.status_code}")
    d = r.json()
    for key in ("team_name", "team_members", "model", "approach", "version"):
        check(f"metadata has {key}", key in d and d[key] not in (None, "", []), str(d))


def test_context_push(client: TestClient) -> None:
    section("context push (idempotency + versioning)")
    cat = load(DATASET / "categories" / "dentists.json")
    merch = load(DATASET / "merchants_seed.json")["merchants"][0]
    trig = load(DATASET / "triggers_seed.json")["triggers"][0]
    cust = load(DATASET / "customers_seed.json")["customers"][0]

    r = client.post("/v1/context", json={
        "scope": "category", "context_id": "dentists", "version": 1,
        "payload": cat, "delivered_at": "2026-04-26T09:45:00Z"})
    check("push category v1 -> 200", r.status_code == 200, f"got {r.status_code} {r.text[:120]}")
    d = r.json()
    check("push returns accepted=true", d.get("accepted") is True, str(d))
    check("push returns ack_id", bool(d.get("ack_id")), str(d))
    check("push returns stored_at", bool(d.get("stored_at")), str(d))

    # The brief: re-posting the same (context_id, version) is idempotent, and
    # 409 is reserved for "you already have a HIGHER version". Getting this
    # wrong makes a judge that re-pushes its opening context mark every
    # context push as FAIL.
    r = client.post("/v1/context", json={
        "scope": "category", "context_id": "dentists", "version": 1,
        "payload": cat, "delivered_at": "2026-04-26T09:45:00Z"})
    check("re-push same version -> 200 (idempotent)", r.status_code == 200,
          f"got {r.status_code} {r.text[:120]}")
    d = r.json()
    check("idempotent re-push returns accepted=true", d.get("accepted") is True, str(d))
    check("idempotent re-push still acks", bool(d.get("ack_id")), str(d))
    check("idempotent re-push does not clobber the stored payload",
          contexts.get("category", "dentists") == cat)

    bumped = json.loads(json.dumps(cat))
    bumped["digest"] = bumped.get("digest", []) + [
        {"id": "d_new", "kind": "research", "title": "NEW injected item", "source": "test"}]
    r = client.post("/v1/context", json={
        "scope": "category", "context_id": "dentists", "version": 2,
        "payload": bumped, "delivered_at": "2026-04-26T10:50:00Z"})
    check("push higher version -> 200", r.status_code == 200, f"got {r.status_code}")
    stored = contexts.get("category", "dentists")
    check("v2 replaced atomically (new digest present)",
          any(i.get("id") == "d_new" for i in stored.get("digest", [])))
    check("old v1 content preserved across replace",
          any(i.get("id") == "d_2026W17_jida_fluoride" for i in stored.get("digest", [])))

    # 409 is for a genuinely OLDER version arriving after a newer one.
    r = client.post("/v1/context", json={
        "scope": "category", "context_id": "dentists", "version": 1,
        "payload": {"digest": []}, "delivered_at": "2026-04-26T11:00:00Z"})
    # v1 < v2 so this must be 409; the digest must survive untouched.
    check("stale (lower) version -> 409", r.status_code == 409,
          f"got {r.status_code} {r.text[:120]}")
    if r.status_code == 409:
        d = r.json()
        check("409 says stale_version", d.get("reason") == "stale_version", str(d))
        check("409 reports current_version=2", d.get("current_version") == 2, str(d))
        check("409 did not overwrite the stored version",
              any(i.get("id") == "d_new"
                  for i in contexts.get("category", "dentists").get("digest", [])))

    r = client.post("/v1/context", json={
        "scope": "merchant", "context_id": merch["merchant_id"], "version": 1, "payload": merch})
    check("push merchant -> 200", r.status_code == 200, f"got {r.status_code}")
    r = client.post("/v1/context", json={
        "scope": "trigger", "context_id": trig["id"], "version": 1, "payload": trig})
    check("push trigger -> 200", r.status_code == 200, f"got {r.status_code}")
    r = client.post("/v1/context", json={
        "scope": "customer", "context_id": cust["customer_id"], "version": 1, "payload": cust})
    check("push customer -> 200", r.status_code == 200, f"got {r.status_code}")

    section("context push (validation)")
    r = client.post("/v1/context", json={
        "scope": "bogus", "context_id": "x", "version": 1, "payload": {}})
    check("bad scope -> 400", r.status_code == 400, f"got {r.status_code}")
    check("bad scope reason == invalid_scope", r.json().get("reason") == "invalid_scope", r.text)

    r = client.post("/v1/context", json={
        "scope": "category", "context_id": "", "version": 1, "payload": {}})
    check("missing context_id -> 400", r.status_code == 400, f"got {r.status_code}")

    r = client.post("/v1/context", content="not json at all")
    check("malformed body -> 400", r.status_code == 400, f"got {r.status_code}")

    r = client.get("/v1/healthz")
    counts = r.json()["contexts_loaded"]
    check("healthz reflects pushed contexts",
          counts == {"category": 1, "merchant": 1, "customer": 1, "trigger": 1}, str(counts))


def test_tick_reply_shapes(client: TestClient) -> None:
    section("tick + reply response shape")
    r = client.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z", "available_triggers": []})
    check("tick returns 200", r.status_code == 200, f"got {r.status_code}")
    check("tick returns actions list", isinstance(r.json().get("actions"), list), r.text)

    r = client.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z",
                                       "available_triggers": ["trg_001_research_digest_dentists"]})
    check("tick with unknown trigger still 200", r.status_code == 200, f"got {r.status_code}")

    r = client.post("/v1/reply", json={
        "conversation_id": "conv_test_1", "from_role": "merchant",
        "message": "Yes please", "received_at": "2026-04-26T10:42:00Z", "turn_number": 2})
    check("reply returns 200", r.status_code == 200, f"got {r.status_code}")
    d = r.json()
    check("reply has action in {send,wait,end}",
          d.get("action") in ("send", "wait", "end"), str(d))
    check("reply has rationale", bool(d.get("rationale")), str(d))

    r = client.post("/v1/reply", json={"message": "hi"})
    check("reply without conversation_id -> 200 + safe action",
          r.status_code == 200 and r.json().get("action") in ("send", "wait", "end"), r.text)


def test_tick_dispatch(client: TestClient) -> None:
    """Real actions, plus the guards that stop us spamming merchants."""
    section("tick dispatch")
    from store import conversations, dispatch

    # Load every category/merchant/trigger from the seed data so the tick has
    # real context to compose against.
    for f in (DATASET / "categories").glob("*.json"):
        cat = load(f)
        client.post("/v1/context", json={"scope": "category", "context_id": cat["slug"],
                                          "version": 1, "payload": cat})
    for m in load(DATASET / "merchants_seed.json")["merchants"]:
        client.post("/v1/context", json={"scope": "merchant",
                                          "context_id": m["merchant_id"],
                                          "version": 1, "payload": m})
    for c in load(DATASET / "customers_seed.json")["customers"]:
        client.post("/v1/context", json={"scope": "customer",
                                          "context_id": c["customer_id"],
                                          "version": 1, "payload": c})
    triggers = load(DATASET / "triggers_seed.json")["triggers"]
    for t in triggers:
        client.post("/v1/context", json={"scope": "trigger", "context_id": t["id"],
                                          "version": 1, "payload": t})

    tids = [t["id"] for t in triggers]
    r = client.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z",
                                      "available_triggers": tids})
    check("tick returns 200", r.status_code == 200, f"got {r.status_code}")
    actions = r.json().get("actions", [])
    check("tick composes actions", len(actions) > 0, f"got {len(actions)}")
    check("tick respects 20-action cap", len(actions) <= 20, f"got {len(actions)}")

    # m_001 is deliberately absent here: test_tick_reply_shapes already ran a
    # tick for trg_001 against the same simulated timestamp, so m_001 is in
    # cooldown and its suppression key is spent. That is the dedup guard doing
    # its job across ticks, not a coverage gap.
    merchant_ids = {a["merchant_id"] for a in actions}
    all_merchants = {m["merchant_id"] for m in load(DATASET / "merchants_seed.json")["merchants"]}
    check("covers every merchant not already messaged",
          merchant_ids == (all_merchants - {"m_001_drmeera_dentist_delhi"}),
          f"got {len(merchant_ids)} merchants")

    required = {"conversation_id", "merchant_id", "send_as", "trigger_id",
                "template_name", "template_params", "body", "cta",
                "suppression_key", "rationale"}
    for a in actions:
        missing = required - set(a)
        check(f"action {a.get('trigger_id', '?')[:28]} has all required fields",
              not missing, f"missing {missing}")
        check(f"action {a.get('trigger_id', '?')[:28]} body is non-empty",
              bool(a.get("body", "").strip()))
        check(f"action {a.get('trigger_id', '?')[:28]} send_as valid",
              a.get("send_as") in ("vera", "merchant_on_behalf"), str(a.get("send_as")))

    bodies = [a["body"] for a in actions]
    check("no URL in any body",
          not any("http://" in b or "https://" in b or "www." in b for b in bodies))
    check("no verbatim duplicate bodies in one tick",
          len(bodies) == len(set(bodies)), "duplicate body in one tick")
    merchants = [a["merchant_id"] for a in actions]
    check("at most one action per merchant per tick",
          len(merchants) == len(set(merchants)), "two actions for one merchant")

    # --- guards ---------------------------------------------------------
    section("tick guards (restraint + dedup)")
    r2 = client.post("/v1/tick", json={"now": "2026-04-26T10:40:00Z",
                                        "available_triggers": tids})
    check("re-tick 5 min later sends nothing new (cooldown + suppression)",
          len(r2.json().get("actions", [])) == 0,
          f"got {len(r2.json().get('actions', []))} — spamming")

    r3 = client.post("/v1/tick", json={"now": "2026-04-28T10:40:00Z",
                                        "available_triggers": tids})
    check("after 2 simulated days some sends resume",
          len(r3.json().get("actions", [])) > 0,
          "suppression never expires")

    r4 = client.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z",
                                        "available_triggers": ["trg_does_not_exist"]})
    check("unknown trigger id -> empty actions, 200",
          r4.status_code == 200 and r4.json().get("actions") == [], r4.text)


def test_no_500(client: TestClient) -> None:
    section("robustness (never 500)")
    cases = [
        ("/v1/context", {}),
        ("/v1/context", {"scope": "merchant"}),
        ("/v1/context", {"scope": "merchant", "context_id": "m", "version": "abc", "payload": {}}),
        ("/v1/tick", {"available_triggers": "not-a-list"}),
        ("/v1/tick", {}),
        ("/v1/reply", {"conversation_id": "c", "message": None}),
        ("/v1/reply", {"conversation_id": "c", "turn_number": "x", "message": "hi"}),
    ]
    for path, body in cases:
        r = client.post(path, json=body)
        check(f"{path} {json.dumps(body)[:44]} -> not 5xx",
              r.status_code < 500, f"got {r.status_code}")


def main() -> int:
    print(f"{YELLOW}{'=' * 62}{RESET}\n  Vera bot — endpoint smoke test\n{YELLOW}{'=' * 62}{RESET}")
    client = TestClient(server.app)

    test_healthz_metadata(client)
    test_context_push(client)
    test_tick_reply_shapes(client)
    test_tick_dispatch(client)
    test_no_500(client)

    total = len(_passed) + len(_failed)
    print(f"\n{YELLOW}{'=' * 62}{RESET}")
    if _failed:
        print(f"  {RED}{len(_failed)}/{total} FAILED{RESET}")
        for name in _failed:
            print(f"    {RED}x{RESET} {name}")
        return 1
    print(f"  {GREEN}ALL {total} CHECKS PASSED{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
