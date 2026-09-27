"""Eyeball every message the shipped judge's context push would produce."""
import json
import logging
import time
from pathlib import Path

from fastapi.testclient import TestClient

import server
from store import conversations, contexts, dispatch

logging.disable(logging.CRITICAL)
D = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")
NOW = "2026-04-26T10:35:00Z"

for s in (contexts, conversations, dispatch):
    s.reset()
c = TestClient(server.app)
for f in (D / "categories").glob("*.json"):
    x = json.loads(f.read_text())
    c.post("/v1/context", json={"scope": "category", "context_id": x["slug"], "version": 1, "payload": x})
mers = json.loads((D / "merchants_seed.json").read_text())["merchants"]
for m in mers[:5]:  # the judge pushes merchants[:5]
    c.post("/v1/context", json={"scope": "merchant", "context_id": m["merchant_id"], "version": 1, "payload": m})
trigs = json.loads((D / "triggers_seed.json").read_text())["triggers"]
for t in trigs:
    c.post("/v1/context", json={"scope": "trigger", "context_id": t["id"], "version": 1, "payload": t})

tids = [t["id"] for t in trigs]
rows, worst = [], []
for i in range(0, len(tids), 5):
    b = tids[i:i + 5]
    t0 = time.time()
    r = c.post("/v1/tick", json={"now": NOW, "available_triggers": b}).json()
    el = time.time() - t0
    print(f"batch {i//5+1}: {len(r.get('actions', []))}/{len(b)} in {el:.1f}s")
    for a in r.get("actions", []):
        rows.append((a["trigger_id"], a["body"], a.get("channel", ""), a.get("to", "")))

print(f"\n{len(rows)}/{len(tids)} answered\n" + "-" * 100)
for tid, body, ch, to in rows:
    print(f"[{tid[:38]:<38s}] {body}")

comp = c.get("/v1/healthz").json().get("composition", {})
print("\ncomposition:", json.dumps(comp))
