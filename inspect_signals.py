#!/usr/bin/env python3
"""Inspect signal selection across the whole seed dataset.

Not a pass/fail test — a review tool. Run it after touching `signals.py` to
see what the bot decided to lead with for every trigger in the dataset, and
whether it stayed grounded.

    .venv/bin/python inspect_signals.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from signals import select_signal

DATASET = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")
NOW = "2026-04-26T10:35:00Z"

GREEN, DIM, CYAN, YELLOW, RED, RESET = (
    "\033[92m", "\033[2m", "\033[96m", "\033[93m", "\033[91m", "\033[0m"
)


def load(filename: str, container: str, key: str) -> dict:
    data = json.loads((DATASET / filename).read_text())
    return {item[key]: item for item in data[container]}


def main() -> int:
    categories = {}
    for f in (DATASET / "categories").glob("*.json"):
        d = json.loads(f.read_text())
        categories[d["slug"]] = d
    merchants = load("merchants_seed.json", "merchants", "merchant_id")
    customers = load("customers_seed.json", "customers", "customer_id")
    triggers = load("triggers_seed.json", "triggers", "id")

    missing_ctx = 0
    placeholder_payloads = 0
    for t in triggers.values():
        pl = t.get("payload") or {}
        if pl.get("placeholder"):
            placeholder_payloads += 1
    if not triggers:
        print("no triggers found")
        return 1

    for tid, t in triggers.items():
        mid = t.get("merchant_id")
        cid = t.get("customer_id")
        m = merchants.get(mid)
        c = customers.get(cid) if cid else None
        if not m:
            missing_ctx += 1
            print(f"{RED}no merchant {mid} for {tid}{RESET}")
            continue
        cat = categories.get(m.get("category_slug", ""), {})

        sig = select_signal(cat, m, t, c, NOW)
        head = f"{CYAN}{tid}{RESET} {DIM}[{t['kind']} u={t.get('urgency')}]{RESET}"
        if sig is None:
            print(f"{head}\n  {RED}NO SIGNAL{RESET}")
            continue

        bar = f"{sig.relevance:.2f}"
        flag = ""
        if not sig.headline:
            flag += f" {RED}[EMPTY HEADLINE]{RESET}"
        if not sig.merchant_link and sig.lever not in ("asking_merchant",):
            flag += f" {YELLOW}[no merchant link]{RESET}"
        print(f"{head} -> {GREEN}{bar}{RESET} {CYAN}{sig.family}{RESET} cta={sig.cta} lever={sig.lever}")
        print(f"    fact : {sig.headline}")
        if sig.citation:
            print(f"    cite : {sig.citation}")
        if sig.merchant_link:
            print(f"    link : {DIM}{sig.merchant_link[:110]}{RESET}")
        if flag:
            print(f"    {flag}")

    print(f"\n{DIM}{len(triggers)} triggers, {missing_ctx} missing merchant ctx, "
          f"{placeholder_payloads} placeholder payloads (not in seeds){RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
