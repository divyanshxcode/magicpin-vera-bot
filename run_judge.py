#!/usr/bin/env python3
"""Run magicpin's official judge_simulator.py against our local bot.

The judge ships with no Nous provider and reads its config from module
constants, so this wrapper does two things without editing their file:

  1. imports judge_simulator as a module
  2. injects a NousProvider that speaks the OpenAI chat-completions shape
     Nous uses, then sets the judge's own LLM_PROVIDER / LLM_API_KEY /
     LLM_MODEL constants before calling main()

Their scoring logic, dataset, prompts and output are untouched — only the
model doing the judging is swapped for one this key can actually reach.

    .venv/bin/python run_judge.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

JUDGE = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/judge_simulator.py")
BOT = os.environ.get("BOT_URL", "http://localhost:8080")
# No default. The judge needs a real key to score with, and a key hardcoded
# here would end up in git history — which is exactly what happened once.
KEY = os.environ.get("LLM_API_KEY", "")
# A judge wants a careful, reasoning-capable model. The bot's own model is
# poolside; the judge is a separate concern and can be stronger.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "poolside/laguna-s-2.1:free")

GREEN, RED, DIM, CYAN, YELLOW, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[2m", "\033[96m", "\033[93m", "\033[1m", "\033[0m"
)


class NousProvider:
    """Drop-in for the judge's LLMProvider interface."""

    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or JUDGE_MODEL
        self.calls = 0

    def name(self) -> str:
        return f"Nous ({self.model})"

    def complete(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1500,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://inference-api.nousresearch.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Cloudflare rejects the default urllib user-agent outright.
                "User-Agent": "magicpin-vera-bot/0.1",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
        self.calls += 1
        msg = data["choices"][0].get("message") or {}
        return msg.get("content") or ""


def main() -> int:
    if not JUDGE.exists():
        print(f"{RED}judge_simulator.py not found at {JUDGE}{RESET}")
        return 1
    if not KEY:
        print(f"{RED}Set LLM_API_KEY{RESET}  (e.g. LLM_API_KEY=... .venv/bin/python run_judge.py)")
        return 1

    # The bot must be up and already holding context, or every score is zero.
    try:
        with urllib.request.urlopen(f"{BOT}/v1/healthz", timeout=10) as r:
            health = json.loads(r.read().decode())
        print(f"{GREEN}bot reachable{RESET}  contexts="
              f"{health.get('contexts_loaded')}")
    except Exception as e:  # noqa: BLE001
        print(f"{RED}bot not reachable at {BOT}{RESET}  {e}")
        return 1

    spec = importlib.util.spec_from_file_location("judge_simulator", JUDGE)
    js = importlib.util.module_from_spec(spec)
    sys.modules["judge_simulator"] = js
    spec.loader.exec_module(js)

    # Register the provider and point the judge's own config at it.
    js.create_provider = lambda: NousProvider(KEY, JUDGE_MODEL)
    js.LLM_PROVIDER = "nous"
    js.LLM_API_KEY = KEY
    js.LLM_MODEL = JUDGE_MODEL
    js.BOT_URL = BOT

    # The judge's scenarios: all (4 behavioural checks), phase2_short,
    # full_evaluation (the scored run over every merchant).
    scenario = os.environ.get("SCENARIO", "full_evaluation")
    print(f"{DIM}scenario={scenario}  judging via {JUDGE_MODEL}{RESET}\n")

    judge = js.JudgeSimulator(NousProvider(KEY, JUDGE_MODEL))
    return 0 if judge.run(scenario) else 1


if __name__ == "__main__":
    sys.exit(main())
