#!/usr/bin/env python3
"""Measure a full 10-action tick per model: sequential vs parallel.

This is the number that actually matters. The judge allows 30s per request and
a tick can carry up to 20 actions, so composition has to fit inside that. One
prompt's latency says nothing; ten in the shape the judge sends them does.

    LLM_API_KEY=... .venv/bin/python bench_tick.py [model ...]
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
import llm
from composer import compose
from store import contexts, conversations, dispatch

DATASET = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")

DEFAULT_MODELS = [
    "poolside/laguna-s-2.1:free",
    "stealth/space-bunny-alpha",
]

GREEN, RED, DIM, CYAN, YELLOW, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[2m", "\033[96m", "\033[93m", "\033[1m", "\033[0m"
)
BUDGET_S = 30.0


def load(p: Path):
    return json.loads(p.read_text())


def seed() -> list[dict]:
    contexts.reset()
    conversations.reset()
    dispatch.reset()
    for f in (DATASET / "categories").glob("*.json"):
        c = load(f)
        contexts.push("category", c["slug"], 1, c)
    for m in load(DATASET / "merchants_seed.json")["merchants"]:
        contexts.push("merchant", m["merchant_id"], 1, m)
    for c in load(DATASET / "customers_seed.json")["customers"]:
        contexts.push("customer", c["customer_id"], 1, c)
    triggers = load(DATASET / "triggers_seed.json")["triggers"]
    for t in triggers:
        contexts.push("trigger", t["id"], 1, t)
    return triggers


def jobs_for(triggers, n=10):
    """One job per distinct merchant — that is what a real tick looks like."""
    from signals import select_signal
    jobs, seen = [], set()
    for t in triggers:
        m = contexts.get("merchant", t["merchant_id"])
        mid = t["merchant_id"]
        if mid in seen:
            continue
        c = contexts.get("customer", t["customer_id"]) if t.get("customer_id") else None
        cat = contexts.get("category", m["category_slug"])
        try:
            sig = select_signal(cat, m, t, c, "2026-04-26T10:35:00Z")
        except Exception:
            continue
        if not sig:
            continue
        jobs.append((t["id"], llm.build_prompt(sig, cat, m, c, None)))
        seen.add(mid)
        if len(jobs) >= n:
            break
    return jobs


def run(model: str, jobs, parallel: bool):
    config.LLM_MODEL = model
    t0 = time.perf_counter()
    if parallel:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            outs = list(pool.map(lambda j: llm.generate(j[1]), jobs))
    else:
        outs = [llm.generate(j[1]) for j in jobs]
    dt = time.perf_counter() - t0
    ok = sum(1 for o in outs if o)
    return dt, ok, outs


def main() -> int:
    if not config.LLM_API_KEY:
        print(f"{RED}Set LLM_API_KEY{RESET}")
        return 1
    models = sys.argv[1:] or DEFAULT_MODELS
    triggers = seed()
    jobs = jobs_for(triggers, 10)

    print(f"{BOLD}{'=' * 74}{RESET}")
    print(f"  {BOLD}Full-tick benchmark{RESET}  {DIM}{len(jobs)} messages per tick, "
          f"budget {BUDGET_S:.0f}s{RESET}")
    print(f"{BOLD}{'=' * 74}{RESET}\n")

    for model in models:
        print(f"{CYAN}{BOLD}{model}{RESET}")
        seq_dt, seq_ok, _ = run(model, jobs, parallel=False)
        par_dt, par_ok, outs = run(model, jobs, parallel=True)

        def verdict(dt, ok, label):
            fit = "FITS" if dt < BUDGET_S else "OVER BUDGET"
            col = GREEN if dt < BUDGET_S else RED
            print(f"    {label:<11} {dt:6.2f}s  ok={ok}/{len(jobs)}  "
                  f"{col}{fit}{RESET}")
            return dt

        verdict(seq_dt, seq_ok, "sequential")
        verdict(par_dt, par_ok, "parallel")
        if seq_ok:
            print(f"    {DIM}per-message {seq_dt / seq_ok:.2f}s sequential, "
                  f"{par_dt / max(par_ok,1):.2f}s parallel{RESET}")
        if par_ok:
            print(f"    {DIM}sample: {outs[0][:100]}{RESET}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
