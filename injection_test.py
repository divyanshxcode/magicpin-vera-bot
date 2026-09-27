#!/usr/bin/env python3
"""Mid-test context injection — the part most bots fail.

From the challenge page:

    "The actual judge harness uses the same scoring logic but injects new facts
     you haven't seen — fresh digest items, performance shifts, surprise
     customer scopes, replies you can't predict. Your score depends on how your
     bot handles those, not on how it does on the 30 pairs. Bots that
     pattern-match the simulator will fail. Bots that ground every output in
     the context they've actually been given will not."

So this simulates Phase 3 of the harness: push a v2 category with a brand-new
digest item, bump a merchant's performance, and add a customer scope that
never existed, then assert the bot actually uses them.

    .venv/bin/python injection_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

import server
from store import contexts, dispatch

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


def push(client: TestClient, scope: str, cid: str, version: int, payload: dict):
    return client.post("/v1/context", json={
        "scope": scope, "context_id": cid, "version": version,
        "payload": payload, "delivered_at": "2026-04-26T11:00:00Z"})


def tick(client: TestClient, now: str, tids: list[str]) -> list[dict]:
    r = client.post("/v1/tick", json={"now": now, "available_triggers": tids})
    return r.json().get("actions", [])


def main() -> int:
    print(f"{BOLD}{'=' * 62}{RESET}\n  Vera bot — mid-test context injection\n{BOLD}{'=' * 62}{RESET}")
    client = TestClient(server.app)

    # ---- Phase 1: base context -----------------------------------------
    dentists = load(DATASET / "categories" / "dentists.json")
    push(client, "category", "dentists", 1, dentists)
    merchants = {m["merchant_id"]: m for m in load(DATASET / "merchants_seed.json")["merchants"]}
    for m in merchants.values():
        push(client, "merchant", m["merchant_id"], 1, m)

    meera = "m_001_drmeera_dentist_delhi"
    base_trigger = next(t for t in load(DATASET / "triggers_seed.json")["triggers"]
                        if t["id"] == "trg_001_research_digest_dentists")

    section("Baseline — compose before any injection")
    push(client, "trigger", base_trigger["id"], 1, base_trigger)
    acts = tick(client, "2026-04-26T10:35:00Z", [base_trigger["id"]])
    check("baseline tick produces an action", len(acts) == 1, f"got {len(acts)}")
    if not acts:
        return 1
    baseline = acts[0]["body"]
    print(f"    {DIM}{baseline[:100]}...{RESET}")

    # ---- Phase 3a: new digest item injected as category v2 -------------
    section("Injection A — new digest item (category v2)")
    updated = json.loads(json.dumps(dentists))
    updated["digest"].insert(0, {
        "id": "d_INJECTED_zirconia_load",
        "kind": "tech",
        "title": "Zirconia framework trial: 34% lower remake rate vs lithium disilicate",
        "source": "JIDA Nov 2026, p.7",
        "trial_n": 940,
        "summary": "Indian multi-centre trial across 22 labs.",
        "actionable": "Ask your lab about zirconia batch numbers before the next crown case",
    })
    r = push(client, "category", "dentists", 2, updated)
    check("category v2 accepted", r.status_code == 200 and r.json().get("accepted"),
          r.text[:80])
    check("store now holds v2", contexts.version("category", "dentists") == 2,
          f"v{contexts.version('category', 'dentists')}")

    injected_trigger = {
        "id": "trg_INJECTED_new_digest", "scope": "merchant",
        "kind": "research_digest", "source": "external",
        "merchant_id": meera, "customer_id": None,
        "payload": {"category": "dentists", "top_item_id": "d_INJECTED_zirconia_load"},
        "urgency": 2, "suppression_key": "research:dentists:2026-W45",
        "expires_at": "2026-05-20T00:00:00Z",
    }
    push(client, "trigger", injected_trigger["id"], 1, injected_trigger)
    acts = tick(client, "2026-04-27T10:35:00Z", [injected_trigger["id"]])
    check("bot composes for the unseen trigger", len(acts) == 1, f"got {len(acts)}")
    if acts:
        body = acts[0]["body"]
        print(f"    {DIM}{body[:100]}...{RESET}")
        check("uses the NEW digest item's headline", "zirconia" in body.lower(), body[:100])
        check("cites the NEW item's source", "JIDA Nov 2026" in body, body[:120])
        check("uses the NEW item's number (940)", "940" in body, body[:120])
        check("does not reuse the old fluoride headline", "fluoride" not in body.lower(), body[:100])

    # ---- Phase 3b: performance shift ----------------------------------
    section("Injection B — merchant performance shifts (v2)")
    m2 = json.loads(json.dumps(merchants[meera]))
    m2["performance"]["ctr"] = 0.041          # was 0.021, now above peer median
    m2["performance"]["calls"] = 47           # was 18
    m2["performance"]["delta_7d"] = {"views_pct": 0.44, "calls_pct": 0.38}
    r = push(client, "merchant", meera, 2, m2)
    check("merchant v2 accepted", r.status_code == 200 and r.json().get("accepted"), r.text[:80])
    check("store holds updated CTR",
          contexts.get("merchant", meera)["performance"]["ctr"] == 0.041)

    perf_trigger = {
        "id": "trg_INJECTED_perf_spike", "scope": "merchant",
        "kind": "perf_spike", "source": "internal",
        "merchant_id": meera, "customer_id": None,
        "payload": {"metric": "calls", "delta_pct": 0.38, "window": "7d",
                    "likely_driver": "new post"},
        "urgency": 2, "suppression_key": "perf_spike:injected:1",
        "expires_at": "2026-05-20T00:00:00Z",
    }
    push(client, "trigger", perf_trigger["id"], 1, perf_trigger)
    acts = tick(client, "2026-04-28T10:35:00Z", [perf_trigger["id"]])
    check("composes for the perf trigger", len(acts) == 1, f"got {len(acts)}")
    if acts:
        body = acts[0]["body"]
        print(f"    {DIM}{body[:110]}...{RESET}")
        check("reflects the NEW 38% delta", "38%" in body, body[:110])
        check("reflects the NEW 47 calls figure", "47" in body, body[:110])
        check("does not quote the STALE 18 calls figure", "18 calls" not in body, body[:110])
        check("does not quote the STALE 2.1% CTR", "2.1%" not in body, body[:110])

    # ---- Phase 3c: surprise customer scope -----------------------------
    section("Injection C — surprise customer scope with a recall trigger")
    new_customer = {
        "customer_id": "c_INJECTED_patient_999",
        "merchant_id": meera,
        "identity": {"name": "Nikhil", "phone_redacted": "<phone>",
                     "language_pref": "hi-en mix", "age_band": "45-55"},
        "relationship": {"first_visit": "2025-08-12", "last_visit": "2025-10-02",
                         "visits_total": 2, "services_received": ["scaling"],
                         "lifetime_value": 1400},
        "state": "lapsed_soft",
        "preferences": {"preferred_slots": "weekday_morning", "channel": "whatsapp",
                        "reminder_opt_in": True},
        "consent": {"opted_in_at": "2025-08-12",
                    "scope": ["recall_reminders", "appointment_reminders"]},
    }
    r = push(client, "customer", new_customer["customer_id"], 1, new_customer)
    check("surprise customer accepted", r.status_code == 200 and r.json().get("accepted"), r.text[:80])

    recall_trigger = {
        "id": "trg_INJECTED_recall_nikhil", "scope": "customer",
        "kind": "recall_due", "source": "internal",
        "merchant_id": meera, "customer_id": new_customer["customer_id"],
        "payload": {"service_due": "scaling_and_polishing",
                    "last_service_date": "2025-10-02", "due_date": "2026-04-02",
                    "available_slots": [
                        {"iso": "2026-04-30T09:00:00+05:30", "label": "Thu 30 Apr, 9am"},
                        {"iso": "2026-05-01T08:00:00+05:30", "label": "Fri 1 May, 8am"}]},
        "urgency": 3, "suppression_key": "recall:injected:nikhil",
        "expires_at": "2026-05-20T00:00:00Z",
    }
    push(client, "trigger", recall_trigger["id"], 1, recall_trigger)
    acts = tick(client, "2026-04-29T10:35:00Z", [recall_trigger["id"]])
    check("composes for the surprise customer scope", len(acts) == 1, f"got {len(acts)}")
    if acts:
        a = acts[0]
        body = a["body"]
        print(f"    {DIM}{body[:130]}...{RESET}")
        check("send_as is merchant_on_behalf",
              a["send_as"] == "merchant_on_behalf", a["send_as"])
        check("uses the new customer's name", "Nikhil" in body, body[:100])
        check("uses the injected slot labels", "30 Apr" in body, body[:130])
        check("uses the injected service", "scaling" in body.lower(), body[:100])
        check("customer_id echoed in the action",
              a["customer_id"] == "c_INJECTED_patient_999", str(a["customer_id"]))
        check("no URL in customer-facing body",
              "http" not in body.lower(), body[:100])

    # ---- unknown / malformed contexts ---------------------------------
    section("Robustness — unknown and malformed injection")
    acts = tick(client, "2026-04-30T10:35:00Z", ["trg_never_pushed_anywhere"])
    check("trigger id never pushed -> no action, no crash", acts == [], f"got {len(acts)}")

    ghost = {"id": "trg_ghost", "scope": "merchant", "kind": "research_digest",
             "source": "external", "merchant_id": "m_does_not_exist",
             "customer_id": None, "payload": {"category": "dentists"},
             "urgency": 2, "suppression_key": "ghost:1",
             "expires_at": "2026-06-01T00:00:00Z"}
    push(client, "trigger", "trg_ghost", 1, ghost)
    acts = tick(client, "2026-04-30T10:35:00Z", ["trg_ghost"])
    check("trigger for an unknown merchant -> no action", acts == [], f"got {len(acts)}")

    empty_cat = push(client, "category", "emptycat", 1, {"slug": "emptycat"})
    check("category with minimal payload accepted",
          empty_cat.status_code == 200, empty_cat.text[:80])

    unknown_kind = {"id": "trg_weird_kind", "scope": "merchant",
                    "kind": "some_kind_from_2031_we_never_saw",
                    "source": "external", "merchant_id": meera, "customer_id": None,
                    "payload": {"placeholder": True}, "urgency": 3,
                    "suppression_key": "weird:1", "expires_at": "2026-06-01T00:00:00Z"}
    push(client, "trigger", unknown_kind["id"], 1, unknown_kind)
    acts = tick(client, "2026-05-01T10:35:00Z", [unknown_kind["id"]])
    check("unseen trigger kind still produces a grounded message", len(acts) == 1,
          f"got {len(acts)}")
    if acts:
        b = acts[0]["body"]
        print(f"    {DIM}{b[:110]}...{RESET}")
        check("unknown-kind message is not empty", bool(b.strip()))
        check("unknown-kind message invents no numbers from the placeholder payload",
              "placeholder" not in b.lower(), b[:100])

    total = len(_passed) + len(_failed)
    print(f"\n{BOLD}{'=' * 62}{RESET}")
    if _failed:
        print(f"  {RED}{len(_failed)}/{total} FAILED{RESET}")
        for n in _failed:
            print(f"    {RED}x{RESET} {n}")
        return 1
    print(f"  {GREEN}ALL {total} INJECTION CHECKS PASSED{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
