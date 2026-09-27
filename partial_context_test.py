"""Regression coverage for partial context.

The shipped judge pushes 5 categories, the first 5 merchants and NO customers
(`list(self.dataset.merchants.items())[:5]`), then offers all 25 triggers. That
is far thinner than what the other suites push, and it used to cost 11 of 25
messages: any trigger naming a merchant that was never pushed hit a bare
`continue`.

This asserts the bot answers from the trigger alone when context is missing,
and — just as important — that it does so without inventing a name.
"""
import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient

import grounding as g
import server
import signals as S
from store import contexts, conversations, dispatch

D = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")
NOW = "2026-04-26T10:35:00Z"

_passed, _failed = [], []


def check(name, cond, detail=""):
    (_passed if cond else _failed).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" {detail}" if not cond and detail else ""))


def main() -> int:
    logging.disable(logging.CRITICAL)
    print("Vera bot — partial context\n" + "=" * 60)

    # --- id parsing -------------------------------------------------------
    print("\n--- Recovering a name from a customer id ---")
    check("a plain customer id yields the name",
          g.customer_name_from_id("c_001_priya_for_m001") == "Priya")
    check("a generational suffix is dropped",
          g.customer_name_from_id("c_012_karthik_jr_for_m008") == "Karthik")
    check("a relationship label is not treated as a name",
          g.customer_name_from_id("c_013_grandfather_for_m009") == "",
          "would greet someone as 'Grandfather'")
    check("an anonymous walk-in is not named",
          g.customer_name_from_id("c_015_anonymous_for_m010") == "")
    for bad in ("", "priya", "c_priya", "x", "c_001", "customer_priya_for_m001"):
        check(f"unfamiliar id {bad!r} yields no name",
              g.customer_name_from_id(bad) == "")

    # --- the judge's thin push -------------------------------------------
    print("\n--- Judge's own context push: 5 categories, 5 merchants, 0 customers ---")
    for s in (contexts, conversations, dispatch):
        s.reset()
    c = TestClient(server.app)
    for f in (D / "categories").glob("*.json"):
        x = json.loads(f.read_text())
        c.post("/v1/context", json={"scope": "category", "context_id": x["slug"],
                                    "version": 1, "payload": x})
    mers = json.loads((D / "merchants_seed.json").read_text())["merchants"]
    for m in mers[:5]:
        c.post("/v1/context", json={"scope": "merchant", "context_id": m["merchant_id"],
                                    "version": 1, "payload": m})
    trigs = json.loads((D / "triggers_seed.json").read_text())["triggers"]
    for t in trigs:
        c.post("/v1/context", json={"scope": "trigger", "context_id": t["id"],
                                    "version": 1, "payload": t})
    tids = [t["id"] for t in trigs]

    got, bodies = set(), {}
    for i in range(0, len(tids), 5):
        r = c.post("/v1/tick", json={"now": NOW,
                                     "available_triggers": tids[i:i + 5]}).json()
        for a in r.get("actions", []):
            got.add(a["trigger_id"])
            bodies[a["trigger_id"]] = a["body"]

    check("every offered trigger gets an answer", len(got) == len(tids),
          f"{len(got)} of {len(tids)}")
    check("triggers for unpushed merchants are answered, not dropped",
          len(got) >= 23, f"only {len(got)} answered")

    # A trigger whose merchant was never pushed must not greet a stranger.
    unpushed = {t["merchant_id"] for t in trigs} - {m["merchant_id"] for m in mers[:5]}
    named = [b for tid, b in bodies.items()
             if (contexts.get("trigger", tid) or {}).get("merchant_id") in unpushed]
    check("messages for unknown merchants do not invent a name",
          all(not b.startswith("Hi,") and not b.startswith("Hi ,") for b in named),
          "found a dangling greeting")
    check("no message opens with a bare 'Hi,'",
          not any(b.startswith("Hi,") for b in bodies.values()))
    check("no message contains the old non-fact 'your 30d'",
          not any("your 30d" in b.lower() for b in bodies.values()))

    # Customer triggers must still reach a real person by name.
    cust_bodies = [b for tid, b in bodies.items()
                   if (contexts.get("trigger", tid) or {}).get("customer_id")]
    check("customer triggers are still addressed to the customer",
          any(b.startswith("Hi Priya") or b.startswith("Hi Kavya")
              or b.startswith("Hi Rashmi") or b.startswith("Hi Karthik")
              for b in cust_bodies),
          "no customer was greeted by their derived name")

    # --- payload-only signals --------------------------------------------
    print("\n--- Payload-only signals ---")
    by_id = {t["id"]: t for t in trigs}
    empty = {"merchant_id": "m_999_unknown_merchant_x"}

    s1 = S.payload_only_signal(empty, by_id["trg_020_summer_demand_shift"], {})
    check("seasonal demand deltas are read out with their signs",
          s1 is not None and "+40" in s1.headline and "-60" in s1.headline,
          f"got {s1.headline if s1 else None!r}")
    s2 = S.payload_only_signal(empty, by_id["trg_021_unverified_gbp_sunrise"], {})
    check("an unverified profile states the size of the prize",
          s2 is not None and "30%" in s2.headline, f"got {s2.headline if s2 else None!r}")
    s3 = S.payload_only_signal(empty, by_id["trg_013_corporate_thali_planning"], {})
    check("a planning trigger quotes the merchant verbatim",
          s3 is not None and "what would it look like" in s3.grounded_quote)
    check("a quoted merchant counts as a checkable fact",
          s3 is not None and S.is_shippable(s3))
    check("a payload with nothing checkable produces no message",
          S.payload_only_signal(empty, {"kind": "some_unknown_kind",
                                        "payload": {"mood": "fine"}}, {}) is None)
    check("a placeholder payload produces no message",
          S.payload_only_signal(empty, {"kind": "x", "payload": {"a": "<placeholder>"}},
                                {}) is None)

    print("\n" + "=" * 60)
    print(f"  {len(_passed)}/{len(_passed) + len(_failed)} checks passed")
    for f in _failed:
        print(f"    x {f}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
