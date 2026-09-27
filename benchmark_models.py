#!/usr/bin/env python3
"""Benchmark candidate models on latency and prose quality.

Latency is a hard constraint: the judge allows 30s per request, and a tick may
need several messages. Sequential composition at 7s each is not viable, so
either a faster model or a parallel path is required. Measure before deciding.

    LLM_API_KEY=... .venv/bin/python benchmark_models.py
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

CANDIDATES = [
    "stealth/space-bunny-alpha",
    "poolside/laguna-s-2.1:free",
    "stepfun/step-3.7-flash:free",
    "upstage/solar-pro4:free",
]

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


def call(model: str, prompt: llm.PromptBundle) -> tuple[str, float, int]:
    """One real call. Returns (text, seconds, prompt_tokens)."""
    key = config.LLM_API_KEY
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": prompt.system},
                     {"role": "user", "content": prompt.user}],
        "temperature": 0.0,
        "max_tokens": 400,
    }
    import httpx
    t0 = time.perf_counter()
    with httpx.Client(timeout=45) as c:
        r = c.post("https://inference-api.nousresearch.com/v1/chat/completions",
                   json=payload,
                   headers={"Authorization": f"Bearer {key}",
                            "content-type": "application/json",
                            "accept": "application/json",
                            "user-agent": llm._DEFAULT_UA})
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return f"HTTP {r.status_code}", dt, 0
    d = r.json()
    text = (d.get("choices") or [{}])[0].get("message", {}).get("content", "")
    usage = d.get("usage") or {}
    return llm.parse_body(text), dt, usage.get("completion_tokens", 0)


def main() -> int:
    if not config.LLM_API_KEY:
        print(f"{RED}Set LLM_API_KEY{RESET}")
        return 1

    triggers = seed()
    # Three real prompts spanning research / compliance / recall, so we see
    # latency on the longest and the shortest shapes.
    picks = ["trg_001_research_digest_dentists",
             "trg_002_compliance_dci_radiograph",
             "trg_003_recall_due_priya"]

    prompts = []
    for tid in picks:
        t = contexts.get("trigger", tid)
        m = contexts.get("merchant", t["merchant_id"])
        c = contexts.get("customer", t["customer_id"]) if t.get("customer_id") else None
        cat = contexts.get("category", m["category_slug"])
        from signals import select_signal
        sig = select_signal(cat, m, t, c, "2026-04-26T10:35:00Z")
        prompts.append((tid, llm.build_prompt(sig, cat, m, c, None)))

    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"  {BOLD}Model benchmark{RESET}  {DIM}3 prompts per model, real calls{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}\n")

    results = {}
    for model in CANDIDATES:
        print(f"{CYAN}{BOLD}{model}{RESET}")
        times, texts, errs = [], [], 0
        for tid, p in prompts:
            text, dt, _ = call(model, p)
            times.append(dt)
            if not text or text.startswith("HTTP"):
                errs += 1
                print(f"  {RED}FAIL{RESET} {tid[:34]:<34} {dt:5.2f}s  "
                      f"{RED}{text[:40]}{RESET}")
            else:
                print(f"  {GREEN} ok {RESET} {tid[:34]:<34} {dt:5.2f}s")
                texts.append((tid, text))
        if times:
            med = statistics.median(times)
            worst = max(times)
            seq10 = med * 10
            print(f"       {DIM}median {med:.2f}s  worst {worst:.2f}s  "
                  f"errors {errs}{RESET}")
            print(f"       {DIM}sequential estimate for a 10-action tick: "
                  f"{seq10:.0f}s{'  ** OVER 30s BUDGET **' if seq10 > 30 else ''}{RESET}")
            results[model] = (med, worst, errs, texts)
        print()

    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"  {BOLD}Ranked by median latency{RESET}")
    for model, (med, worst, errs, _) in sorted(
            results.items(), key=lambda kv: kv[1][0]):
        flag = "over budget" if med * 10 > 30 else "ok"
        print(f"    {med:6.2f}s  err={errs}  seq10={med*10:6.0f}s [{flag}]  {model}")

    print(f"\n{BOLD}Sample output per model{RESET}\n")
    for model, (_, _, errs, texts) in results.items():
        if not texts:
            print(f"{RED}{model} — no successful outputs{RESET}\n")
            continue
        tid, text = texts[0]
        print(f"{CYAN}{model}{RESET}  {DIM}({tid}){RESET}")
        print(f"  {text}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
