#!/usr/bin/env python3
"""Compose the seed dataset with the REAL model and review the output.

The stubbed seam test proves the plumbing. This proves the prose — which is
the part actually being scored. It prints the model's messages next to the
deterministic baseline so the difference is visible side by side.

    LLM_PROVIDER=nous LLM_API_KEY=... LLM_MODEL=... .venv/bin/python live_llm_test.py [n]
"""

from __future__ import annotations

import config
import json
import sys
import time
from pathlib import Path

import grounding as g
from composer import compose, compose_body
from signals import select_signal
from store import contexts, conversations, dispatch

DATASET = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")

GREEN, RED, DIM, CYAN, YELLOW, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[2m", "\033[96m", "\033[93m", "\033[1m", "\033[0m"
)


def load(p: Path):
    return json.loads(p.read_text())


def seed() -> list[dict]:
    contexts.reset()
    conversations.reset()
    dispatch.reset()
    for f in (DATASET / "categories").glob("*.json"):
        cat = load(f)
        contexts.push("category", cat["slug"], 1, cat)
    for m in load(DATASET / "merchants_seed.json")["merchants"]:
        contexts.push("merchant", m["merchant_id"], 1, m)
    for c in load(DATASET / "customers_seed.json")["customers"]:
        contexts.push("customer", c["customer_id"], 1, c)
    triggers = load(DATASET / "triggers_seed.json")["triggers"]
    for t in triggers:
        contexts.push("trigger", t["id"], 1, t)
    return triggers


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    if not config.llm_configured():
        print(f"{RED}LLM not configured. Set LLM_PROVIDER / LLM_API_KEY / LLM_MODEL.{RESET}")
        return 1

    print(f"{BOLD}{'=' * 74}{RESET}")
    print(f"  {BOLD}Live composition{RESET}  "
          f"{DIM}{config.LLM_PROVIDER}:{config.LLM_MODEL} "
          f"temp={config.LLM_TEMPERATURE}{RESET}")
    print(f"{BOLD}{'=' * 74}{RESET}")

    triggers = seed()
    ok = fail = 0
    timings: list[float] = []
    total_cost = 0.0

    for t in triggers[:limit]:
        trig = contexts.get("trigger", t["id"])
        merchant = contexts.get("merchant", trig["merchant_id"])
        customer = (contexts.get("customer", trig["customer_id"])
                    if trig.get("customer_id") else None)
        category = contexts.get("category", merchant["category_slug"])

        t0 = time.perf_counter()
        out = compose(category, merchant, trig, customer, "2026-04-26T10:35:00Z")
        ms = (time.perf_counter() - t0) * 1000
        timings.append(ms)

        if not out:
            print(f"\n{RED}no message{RESET} {t['id']}  ({ms:.0f}ms)")
            fail += 1
            continue

        sig = select_signal(category, merchant, trig, customer, "2026-04-26T10:35:00Z")
        det = compose_body(sig, category, merchant, customer) if sig else ""
        used_llm = out["body"] != det

        body = out["body"]
        if "http" in body.lower():
            print(f"\n{RED}URL LEAKED{RESET} {t['id']}")
            fail += 1
            continue
        if len(body) > 400:
            print(f"\n{RED}TOO LONG{RESET} ({len(body)}) {t['id']}")
            fail += 1
            continue

        ok += 1
        tag = f"{GREEN}LLM{RESET}" if used_llm else f"{YELLOW}det{RESET}"
        print(f"\n{CYAN}{BOLD}{t['id']}{RESET}  {tag} {DIM}{ms:.0f}ms{RESET}  "
              f"{DIM}{out['cta']}{RESET}")
        print(f"  {BOLD}{body}{RESET}")
        if not used_llm:
            print(f"  {DIM}deterministic fallback (model call failed or returned nothing){RESET}")
        else:
            print(f"  {DIM}baseline was: {det[:96]}{RESET}")

    n = len(timings)
    if n:
        print(f"\n{BOLD}{'=' * 74}{RESET}")
        print(f"  {GREEN}{ok} composed{RESET}"
              + (f", {RED}{fail} rejected{RESET}" if fail else "")
              + f"   median {sorted(timings)[n // 2]:.0f}ms"
              + f"   max {max(timings):.0f}ms")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
