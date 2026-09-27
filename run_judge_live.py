#!/usr/bin/env python3
"""Run the official judge against the DEPLOYED service.

Same script as run_judge.py, but pointed at the public URL and with timeouts
sized for Render's free tier, where a cold start can exceed the judge's own
30s budget. The bot is remote now, so /v1/tick is a network round trip plus
composition rather than a local call.

    BOT_URL=https://your-service.onrender.com \
    JUDGE_MODEL=gemini-3.1-flash-lite \
    LLM_API_KEY=... .venv/bin/python run_judge_live.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_judge import JUDGE, KEY, JUDGE_MODEL  # noqa: E402

BASE = os.environ.get("BOT_URL", "https://magicpin-vera-bot-hmqb.onrender.com").rstrip("/")
SCENARIO = os.environ.get("SCENARIO", "full_evaluation")

GREEN, RED, DIM, YELLOW, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[2m", "\033[93m", "\033[1m", "\033[0m"
)


def main() -> int:
    if not KEY:
        print(f"{RED}Set LLM_API_KEY{RESET}")
        return 1

    # The deployed bot must be awake. Render free spins down after 15 minutes
    # idle, and a cold start can take longer than the judge's request timeout,
    # so warm it before the run rather than losing batch 1.
    print(f"{DIM}warming {BASE} …{RESET}")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{BASE}/v1/healthz", timeout=120) as r:
            json.loads(r.read().decode())
        print(f"{GREEN}awake{RESET} in {time.perf_counter() - t0:.1f}s\n")
    except Exception as e:  # noqa: BLE001
        print(f"{RED}not reachable: {e}{RESET}")
        return 1

    spec = importlib.util.spec_from_file_location("judge_simulator", JUDGE)
    js = importlib.util.module_from_spec(spec)
    sys.modules["judge_simulator"] = js
    spec.loader.exec_module(js)

    # Wipe state first. The bot suppresses a trigger for 24h after sending it,
    # so a service that has already ticked returns {actions: []} and the judge
    # scores nothing. /v1/teardown exists for exactly this.
    try:
        req = urllib.request.Request(f"{BASE}/v1/teardown", method="POST",
                                     data=b"{}",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            print(f"{DIM}teardown: {r.read().decode()[:80]}{RESET}")
    except Exception as e:  # noqa: BLE001
        print(f"{DIM}teardown not available ({type(e).__name__}) — "
              f"already-committed sends stay suppressed{RESET}")

    # Use the judge's OWN provider factory. Pointing it at a Nous endpoint with
    # a Gemini model name returns 404 on every call, which silently scores each
    # message as a flat 5/10 across all five dimensions.
    js.LLM_PROVIDER = os.environ.get("JUDGE_PROVIDER", "gemini")
    js.LLM_API_KEY = KEY
    js.LLM_MODEL = JUDGE_MODEL
    js.BOT_URL = BASE

    # The judge hardcodes 15s at each BotClient call site. Against a remote host
    # on a free tier that is too tight — a full 10-action tick measures 7-12s.
    _orig_request = js.BotClient._request

    def patient_request(self, method, path, timeout=90, body_dict=None):
        return _orig_request(self, method, path, 120, body_dict)

    js.BotClient._request = patient_request

    print(f"{DIM}scenario={SCENARIO}  judge={js.LLM_PROVIDER}/{JUDGE_MODEL}{RESET}\n")
    judge = js.JudgeSimulator(js.create_provider())
    ok = judge.run(SCENARIO)

    # Distinguish real scores from the judge's own fallbacks. _fallback_score
    # marks every dimension "Could not evaluate", so a message scored that way
    # says nothing about the bot and would drag the average down unfairly.
    real = [s for s in judge.all_scores if "Could not evaluate" not in s.specificity_reason]
    if real and len(real) != len(judge.all_scores):
        n, d = len(real), len(judge.all_scores)
        avg = sum(s.total for s in real) // len(real)
        print(f"\n{YELLOW}  {n} of {d} messages were genuinely scored; "
              f"{d - n} fell back to the judge's default 5s (429s / JSON "
              f"parse failures on the judge side).{RESET}")
        print(f"{YELLOW}  Mean of the real scores: {avg}/50{RESET}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
