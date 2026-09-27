#!/usr/bin/env python3
"""Model benchmark: screen candidates, then time the survivors.

Latency is a hard constraint — the judge allows 30s per request and /v1/tick
composes up to 20 messages inside one response. So the number that matters is a
full tick, not one prompt.

Candidates are screened first with a single real prompt, because several look
fine in a model list and are not. Measured on this key:

  poolside/laguna-s-2.1:free            10/10   11.6s parallel   <- current
  inclusionai/ling-3.0-flash-fin:free     7/10    8.0s parallel
  inclusionai/ling-3.0-flash-sante:free   0/10   burns the whole token budget
  meituan/longcat-2.0:free               404    retired from :free
  meituan/longcat-2.0                   404    paid, needs a balance
  stepfun/step-3.7-flash:free            400    "missing tags", upstream issue
  upstage/solar-pro4:free                400    "missing tags", upstream issue

ling-fin is genuinely faster but rate-limits under 10-way concurrency (0/10 at
full burst), so its apparent speed is paid for in silent fallbacks.

    LLM_API_KEY=... .venv/bin/python model_bench.py
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
NOW = "2026-04-26T10:35:00Z"
BUDGET_S = 30.0

SCREEN = [
    "poolside/laguna-s-2.1:free",
    "inclusionai/ling-3.0-flash-fin:free",
    "inclusionai/ling-3.0-flash-sante:free",
    "meituan/longcat-2.0",
    "stealth/space-bunny-alpha",
]

GREEN, RED, DIM, CYAN, YELLOW, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[2m", "\033[96m", "\033[93m", "\033[1m", "\033[0m"
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


def one_raw(model, effort, prompt):
    """One direct API call, bypassing the cache. Returns (text, seconds)."""
    import http.client
    p = {"model": model, "temperature": 0.0, "max_tokens": config.LLM_MAX_TOKENS,
         "messages": [{"role": "system", "content": prompt.system},
                      {"role": "user", "content": prompt.user}]}
    if effort:
        p["reasoning_effort"] = effort
    t0 = time.perf_counter()
    try:
        c = http.client.HTTPSConnection("inference-api.nousresearch.com", timeout=90)
        c.request("POST", "/v1/chat/completions", json.dumps(p), {
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": llm._DEFAULT_UA, "Accept": "application/json"})
        raw = c.getresponse().read().decode()
        dt = time.perf_counter() - t0
        d = json.loads(raw)
        txt = ((d["choices"][0].get("message") or {}).get("content") or "").strip()
        return txt, dt
    except Exception:
        return "", time.perf_counter() - t0


def main() -> int:
    if not config.LLM_API_KEY:
        print(f"{RED}Set LLM_API_KEY{RESET}")
        return 1
    seed()

    from signals import select_signal
    prompts = []
    seen = set()
    for t in load(DATASET / "triggers_seed.json")["triggers"]:
        if t["merchant_id"] in seen:
            continue
        m = contexts.get("merchant", t["merchant_id"])
        c = contexts.get("customer", t["customer_id"]) if t.get("customer_id") else None
        cat = contexts.get("category", m["category_slug"])
        sig = select_signal(cat, m, t, c, NOW)
        if sig:
            prompts.append(llm.build_prompt(sig, cat, m, c, None))
            seen.add(t["merchant_id"])
        if len(prompts) == 10:
            break

    print(f"{BOLD}{'=' * 76}{RESET}")
    print(f"  {BOLD}Screen{RESET}  {DIM}1 real prompt per model{RESET}")
    print(f"{BOLD}{'=' * 76}{RESET}\n")

    viable = []
    for model in SCREEN:
        txt, dt = one_raw(model, None, prompts[0])
        usable = bool(txt) and len(txt) > 20
        mark = f"{GREEN}USABLE{RESET}" if usable else f"{RED}unusable{RESET}"
        print(f"  {mark}  {model:<42} {dt:5.2f}s  "
              f"{DIM}{(txt[:58] if txt else '(empty response)')}{RESET}")
        if usable:
            viable.append(model)
        time.sleep(0.3)

    print(f"\n{BOLD}{'=' * 76}{RESET}")
    print(f"  {BOLD}Full tick{RESET}  {DIM}{len(prompts)} messages, "
          f"sequential vs parallel, budget {BUDGET_S:.0f}s{RESET}")
    print(f"{BOLD}{'=' * 76}{RESET}\n")

    rows = []
    for model in viable:
        with llm._CACHE_LOCK:
            llm._CACHE.clear()
        config.LLM_MODEL = llm.config.LLM_MODEL = model

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            outs = list(pool.map(lambda p: llm.generate(p), prompts))
        par = time.perf_counter() - t0
        ok = sum(1 for o in outs if o)

        # Sequential cost, from 3 direct calls.
        seq3 = [one_raw(model, None, p)[1] for p in prompts[:3]]
        seq_est = statistics.median(seq3) * len(prompts)

        col = GREEN if par < BUDGET_S else RED
        print(f"{CYAN}{BOLD}{model}{RESET}")
        print(f"    parallel   {par:6.2f}s  {col}{'FITS' if par < BUDGET_S else 'OVER BUDGET'}{RESET}")
        print(f"    sequential {seq_est:6.2f}s  (median x {len(prompts)})  "
              f"{RED}{'would blow the budget' if seq_est > BUDGET_S else 'ok'}{RESET}")
        print(f"    success    {ok}/{len(prompts)}")
        if ok:
            print(f"    {DIM}{outs[0][:130]}{RESET}")
            empt = sum(1 for o in outs if not o)
            if empt:
                print(f"    {YELLOW}{empt} fell back to deterministic{RESET}")
        rows.append((model, par, seq_est, ok, len(prompts)))
        print()

    if rows:
        print(f"{BOLD}{'=' * 76}{RESET}")
        print(f"  {BOLD}Verdict{RESET}  (ranked by success, then speed)")
        for model, par, seq, ok, n in sorted(rows, key=lambda r: (-r[3], r[1])):
            print(f"    {ok}/{n}  parallel {par:5.2f}s  seq~{seq:6.1f}s   {model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
