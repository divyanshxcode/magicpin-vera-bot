"""Consent is honoured, and the refusal is visible.

`consent_allows` shipped in grounding.py from the first commit and nothing
ever called it. Consent in this dataset is per-topic and explicit: c_015's
scope is an empty list, c_008 consented to promotional offers and nothing
else, c_012 to kids program updates alone. Every one of those customers was
being messaged about subjects they had not agreed to hear about.

The risk in fixing that is over-blocking, so both directions are asserted:
a refusal where consent is absent, and a send everywhere else — including
when the judge pushes no customer records at all, which is what it actually
does.
"""
import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient

import server
import tick
from store import contexts, conversations, dispatch

D = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")
NOW = "2026-04-26T10:35:00Z"

_passed, _failed = [], []


def check(name, cond, detail=""):
    (_passed if cond else _failed).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if not cond and detail else ""))


def cust(pref_name, scope, cid="c_x"):
    return {"customer_id": cid,
            "identity": {"name": pref_name, "language_pref": "english"},
            "state": "active",
            "relationship": {"last_visit": "2026-01-12", "visits_total": 4},
            "consent": {"scope": scope}}


def trig(kind, cid="c_x"):
    return {"id": "t", "kind": kind, "scope": "customer", "customer_id": cid,
            "merchant_id": "m_001", "urgency": 3, "payload": {}}


def main() -> int:
    logging.disable(logging.CRITICAL)
    print("Vera bot — consent\n" + "=" * 60)

    print("\n--- refusals ---")
    check("an empty scope refuses everything in scope",
          not tick.customer_consent_ok(trig("recall_due"),
                                       cust("Anon", [])))
    check("promotional-only consent does not cover a recall reminder",
          not tick.customer_consent_ok(trig("recall_due"),
                                       cust("Dipti", ["promotional_offers"])))
    check("promotional-only consent does not cover a trial follow-up",
          not tick.customer_consent_ok(trig("trial_followup"),
                                       cust("Dipti", ["promotional_offers"])))
    check("kids-only consent does not cover a refill",
          not tick.customer_consent_ok(trig("chronic_refill_due"),
                                       cust("Karthik Jr", ["kids_program_updates"])))

    print("\n--- sends ---")
    check("matching consent sends",
          tick.customer_consent_ok(trig("recall_due"),
                                   cust("Priya", ["recall_reminders",
                                                  "appointment_reminders"])))
    check("a broader scope covers a narrower subject",
          tick.customer_consent_ok(trig("recall_due"),
                                   cust("Rohit", ["recall_reminders",
                                                  "appointment_reminders",
                                                  "treatment_followup"])))
    check("bridal consent covers the bridal follow-up",
          tick.customer_consent_ok(trig("wedding_package_followup"),
                                   cust("Kavya", ["appointment_reminders",
                                                  "bridal_package_followup"])))
    check("a customer record with no consent block is still messaged",
          tick.customer_consent_ok(trig("recall_due"),
                                   {"customer_id": "c_x", "identity": {"name": "X"}}))
    check("an unknown trigger kind is not blocked",
          tick.customer_consent_ok(trig("some_brand_new_kind"),
                                   cust("Dipti", ["promotional_offers"])))
    check("a merchant-facing trigger is never consent-gated",
          tick.customer_consent_ok({"id": "t", "kind": "perf_dip",
                                    "merchant_id": "m_001", "payload": {}}, None))
    check("a merchant-facing trigger is not gated by a customer's scope",
          tick.customer_consent_ok({"id": "t", "kind": "perf_dip",
                                    "merchant_id": "m_001", "payload": {},
                                    "customer_id": "c_x"},
                                   cust("Dipti", [])))

    print("\n--- end to end through /v1/tick ---")
    for s in (contexts, conversations, dispatch):
        s.reset()
    c = TestClient(server.app)
    mers = json.loads((D / "merchants_seed.json").read_text())["merchants"]
    for m in mers[:2]:
        c.post("/v1/context", json={"scope": "merchant", "context_id": m["merchant_id"],
                                    "version": 1, "payload": m})
    cat = json.loads((D / "categories/dentists.json").read_text())
    c.post("/v1/context", json={"scope": "category", "context_id": cat["slug"],
                                "version": 1, "payload": cat})

    no = cust("Dipti", ["promotional_offers"], "c_no")
    yes = cust("Priya", ["recall_reminders"], "c_yes")
    for cu in (no, yes):
        c.post("/v1/context", json={"scope": "customer", "context_id": cu["customer_id"],
                                    "version": 1, "payload": cu})
    t_no = {"id": "t_no", "kind": "recall_due", "scope": "customer",
            "customer_id": "c_no", "merchant_id": mers[0]["merchant_id"], "urgency": 3,
            "suppression_key": "sk_no", "payload": {"service_due": "6_month_cleaning",
                                                    "due_date": "2026-07-12"}}
    t_yes = dict(t_no, id="t_yes", customer_id="c_yes", suppression_key="sk_yes")
    for t in (t_no, t_yes):
        c.post("/v1/context", json={"scope": "trigger", "context_id": t["id"],
                                    "version": 1, "payload": t})

    r = c.post("/v1/tick", json={"now": NOW,
                                 "available_triggers": ["t_no", "t_yes"]}).json()
    sent = {a["customer_id"] for a in r.get("actions", [])}
    check("the customer who did not consent is not messaged", "c_no" not in sent, str(sent))
    check("the customer who did consent is messaged", "c_yes" in sent, str(sent))
    check("the refusal is recorded rather than silent",
          "c_no" in dispatch.consent_blocked, str(dispatch.consent_blocked))

    h = c.get("/v1/healthz").json()
    check("healthz survives the consent state", h.get("status") == "ok")
    check("teardown clears the consent log",
          (c.post("/v1/teardown").status_code < 400
           and dispatch.consent_blocked == []))

    print("\n" + "=" * 60)
    print(f"  {len(_passed)}/{len(_passed) + len(_failed)} checks passed")
    for f in _failed:
        print(f"    x {f}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
