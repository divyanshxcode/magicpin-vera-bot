#!/usr/bin/env python3
"""Print every composed message for the seed dataset.

The signal inspector showed what we decided to *say*; this shows the finished
message. Read these against the 10 case studies in examples/case-studies.md —
a 50/50 anchor looks like Case Study 1:

    "Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk
     adult patients — 2,100-patient trial showed 3-month fluoride recall cuts
     caries recurrence 38% better than 6-month. Worth a look (2-min abstract).
     Want me to pull it + draft a patient-ed WhatsApp you can share?"

    .venv/bin/python inspect_messages.py
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from composer import compose

DATASET = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")
NOW = "2026-04-26T10:35:00Z"

GREEN, DIM, CYAN, YELLOW, RED, BOLD, RESET = (
    "\033[92m", "\033[2m", "\033[96m", "\033[93m", "\033[91m", "\033[1m", "\033[0m"
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

    only = sys.argv[1] if len(sys.argv) > 1 else ""
    flags: list[str] = []
    n_ok = 0

    for tid, t in triggers.items():
        if only and only not in tid:
            continue
        m = merchants.get(t.get("merchant_id"))
        c = customers.get(t.get("customer_id")) if t.get("customer_id") else None
        if not m:
            continue
        cat = categories.get(m.get("category_slug", ""), {})

        out = compose(cat, m, t, c, NOW)
        head = f"{BOLD}{tid}{RESET} {DIM}[{t['kind']}]{RESET}"

        if out is None:
            print(f"{head}\n  {YELLOW}restraint: nothing worth sending{RESET}")
            continue

        n_ok += 1
        print(f"\n{head} {CYAN}{out['send_as']}{RESET} {DIM}cta={out['cta']}{RESET}")
        body = out["body"]
        wrapped = textwrap.fill(body, width=84, initial_indent="  ", subsequent_indent="  ")
        print(wrapped)
        if flags:
            print(f"  {RED}FLAGS:{RESET} {flags}")
        flags = []
        print(f"  {DIM}why: {out['rationale'][:150]}{RESET}")

    print(f"\n{DIM}{'=' * 70}")
    print(f"  {n_ok} messages composed from {len(triggers)} seed triggers")
    print(f"  anchors to compare: examples/case-studies.md (10 cases, 44-50/50)")
    print(f"{'=' * 70}{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
