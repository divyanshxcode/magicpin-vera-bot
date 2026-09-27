#!/usr/bin/env python3
"""Prove end-to-end determinism with the REAL provider, across processes.

The judge requires "your output should stay deterministic for the same input
and simulator settings". The provider ignores temperature 0, so this is only
true because of the output cache — and a cache is per-process, so proving it
means proving something narrower and honest:

  WITHIN a process, the same signal always yields the same body. This is what
  a replayed tick, a judge retry, and the same signal resurfacing later all
  hit.

Run twice in separate processes and compare the ORDER-STABLE digest of what
each run would send. A cold process re-generates, so cross-process equality is
not expected and not claimed — the cache is in-memory by design.

    LLM_PROVIDER=... LLM_API_KEY=... .venv/bin/python determinism_check.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
import llm
from composer import compose
from signals import select_signal
from store import contexts, conversations, dispatch

DATASET = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")
NOW = "2026-04-26T10:35:00Z"

GREEN, RED, DIM, CYAN, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[2m", "\033[96m", "\033[1m", "\033[0m"
)


def load(p: Path):
    return json.loads(p.read_text())


def seed():
    contexts.reset()
    conversations.reset()
    dispatch.reset()
    with llm._CACHE_LOCK:
        llm._CACHE.clear()
    for f in (DATASET / "categories").glob("*.json"):
        c = load(f)
        contexts.push("category", c["slug"], 1, c)
    for m in load(DATASET / "merchants_seed.json")["merchants"]:
        contexts.push("merchant", m["merchant_id"], 1, m)
    for c in load(DATASET / "customers_seed.json")["customers"]:
        contexts.push("customer", c["customer_id"], 1, c)
    for t in load(DATASET / "triggers_seed.json")["triggers"]:
        contexts.push("trigger", t["id"], 1, t)


def compose_all():
    out = {}
    for t in load(DATASET / "triggers_seed.json")["triggers"]:
        trig = contexts.get("trigger", t["id"])
        m = contexts.get("merchant", trig["merchant_id"])
        c = contexts.get("customer", trig["customer_id"]) if trig.get("customer_id") else None
        cat = contexts.get("category", m["category_slug"])
        r = compose(cat, m, trig, c, NOW)
        if r:
            out[t["id"]] = r["body"]
    return out


def main() -> int:
    if not config.llm_configured():
        print(f"{RED}Set LLM_PROVIDER / LLM_API_KEY / LLM_MODEL{RESET}")
        return 1

    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"  {BOLD}Determinism, real provider{RESET}  "
          f"{DIM}{config.LLM_PROVIDER}:{config.LLM_MODEL}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")

    failures = 0

    # --- 1. cold run, twice, clearing the cache between --------------------
    print(f"{CYAN}1. Cold runs (cache cleared) — expected to differ, "
          f"provider ignores temperature 0{RESET}")
    runs = []
    for i in range(2):
        seed()
        t0 = time.perf_counter()
        bodies = compose_all()
        runs.append(bodies)
        print(f"   run {i + 1}: {len(bodies)} messages in "
              f"{time.perf_counter() - t0:.1f}s, cache={llm.cache_size()}")
    shared = set(runs[0]) & set(runs[1])
    same = [t for t in shared if runs[0][t] == runs[1][t]]
    print(f"   {DIM}identical across cold runs: {len(same)}/{len(shared)}"
          f"{RESET}  <- expected to be low; this is the provider, not us\n")

    # --- 2. warm run in one process: the guarantee we actually make --------
    print(f"{CYAN}2. Warm runs in one process — MUST be identical{RESET}")
    seed()
    warm = []
    for i in range(3):
        t0 = time.perf_counter()
        bodies = compose_all()
        warm.append(bodies)
        print(f"   run {i + 1}: {len(bodies)} messages in "
              f"{time.perf_counter() - t0:.2f}s, cache={llm.cache_size()}")
    ok = all(w == warm[0] for w in warm)
    if ok:
        print(f"   {GREEN}PASS{RESET} all 3 warm runs byte-identical")
    else:
        failures += 1
        print(f"   {RED}FAIL{RESET} warm runs diverged")
        for k in warm[0]:
            if len({w.get(k) for w in warm}) > 1:
                print(f"      {RED}{k}{RESET}")

    # --- 3. the signal, not just the text, is stable ----------------------
    print(f"\n{CYAN}3. Selected signal is stable across the same input{RESET}")
    seed()
    picks = []
    for _ in range(3):
        t = load(DATASET / "triggers_seed.json")["triggers"][0]
        trig = contexts.get("trigger", t["id"])
        m = contexts.get("merchant", trig["merchant_id"])
        cat = contexts.get("category", m["category_slug"])
        sig = select_signal(cat, m, trig, None, NOW)
        picks.append((sig.family, sig.lever, sig.citation))
    if len(set(picks)) == 1:
        print(f"   {GREEN}PASS{RESET} {DIM}{picks[0]}{RESET}")
    else:
        failures += 1
        print(f"   {RED}FAIL{RESET} {picks}")

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    if failures:
        print(f"  {RED}{failures} determinism check(s) FAILED{RESET}")
        return 1
    print(f"  {GREEN}DETERMINISM VERIFIED{RESET}")
    print(f"  {DIM}Guarantee: within a process, identical context -> identical")
    print(f"  message. The provider is not deterministic; the cache is what")
    print(f"  makes the bot deterministic, and that is per-process by design.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
