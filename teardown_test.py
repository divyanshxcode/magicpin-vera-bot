#!/usr/bin/env python3
"""Teardown must leave the bot usable, not bricked.

Regression test. /v1/teardown used to rebind store.contexts/conversations/
dispatch to fresh objects. Modules that do `from store import contexts` at
import time — tick.py and reply_handler.py both do — keep the old reference, so
after a teardown the judge could push context, /v1/healthz would report it
loaded, and /v1/tick would return {"actions": []} in under half a second,
permanently, with no error anywhere.

That is exactly what happened on the deployed service.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

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


def seed(c: TestClient, n_merchants: int = 4) -> list[str]:
    for f in (DATASET / "categories").glob("*.json"):
        x = json.loads(f.read_text())
        c.post("/v1/context", json={"scope": "category", "context_id": x["slug"],
                                    "version": 1, "payload": x})
    merchants = json.loads((DATASET / "merchants_seed.json").read_text())["merchants"]
    for m in merchants[:n_merchants]:
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
        if len(picked) == n_merchants:
            break
    return picked


def main() -> int:
    for s in (contexts, conversations, dispatch):
        s.reset()
    c = TestClient(server.app)

    print(f"{BOLD}Teardown must leave the service usable{RESET}\n")

    # 1. Baseline: works before any teardown.
    ids = seed(c)
    r = c.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z",
                                 "available_triggers": ids}).json()
    before = len(r.get("actions", []))
    check("tick works before teardown", before > 0, f"{before} actions")

    # 2. Teardown.
    r = c.post("/v1/teardown")
    check("teardown returns 200", r.status_code == 200, str(r.status_code))
    check("teardown reports wiped", r.json().get("wiped") is True, str(r.json()))

    # 3. The critical one: state is actually gone.
    h = c.get("/v1/healthz").json()["contexts_loaded"]
    check("contexts actually wiped", sum(h.values()) == 0, str(h))

    # 4. The bug: does tick still see freshly-pushed context?
    ids = seed(c)
    r = c.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z",
                                 "available_triggers": ids}).json()
    after = len(r.get("actions", []))
    check("tick still works AFTER teardown", after > 0, f"{after} actions")
    check("post-teardown count matches pre-teardown", after == before,
          f"{before} -> {after}")

    # 5. Reply path too — reply_handler.py holds its own store reference.
    if after:
        a = r["actions"][0]
        rr = c.post("/v1/reply", json={
            "conversation_id": a["conversation_id"],
            "merchant_id": a["merchant_id"],
            "merchant_message": "Not interested.", "turn_number": 1})
        check("reply works after teardown", rr.status_code == 200
              and rr.json().get("action") in ("send", "wait", "end"),
              str(rr.json().get("action")))

    # 6. Two teardowns in a row, then work again.
    c.post("/v1/teardown")
    c.post("/v1/teardown")
    ids = seed(c)
    r = c.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z",
                                 "available_triggers": ids}).json()
    check("works after two consecutive teardowns",
          len(r.get("actions", [])) > 0, f"{len(r.get('actions', []))} actions")

    # 7. Suppression is genuinely cleared — a second identical tick is empty.
    r2 = c.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z",
                                  "available_triggers": ids}).json()
    check("suppression still applies after teardown (not over-cleared)",
          len(r2.get("actions", [])) == 0, f"{len(r2.get('actions', []))} actions")

    print(f"\n  {GREEN if passed == total else RED}{passed}/{total} checks passed{RESET}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
