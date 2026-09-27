#!/usr/bin/env python3
"""Run the judge's full_evaluation with the per-dimension rationales shown.

`run_judge.py` executes the judge's scenarios as shipped, and `_full` calls
`_score_and_display(..., verbose=False)` — which prints the five scores and
drops the reasoning. The reasoning is where the actual diagnosis lives: a 23/50
message says nothing on its own, but "fabricated a countdown the payload
contradicts" says exactly what to fix.

The judge is loaded as a module and only ONE thing is changed — the display
verbosity of the scoring loop. Their dataset, prompt, scoring, and output are
untouched, so the numbers printed are the numbers their simulator produces.

    LLM_API_KEY=... JUDGE_MODEL=... .venv/bin/python run_judge_verbose.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

JUDGE = Path(os.environ.get(
    "JUDGE_PATH",
    "/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/judge_simulator.py"))
KEY = os.environ.get("LLM_API_KEY", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "poolside/laguna-s-2.1:free")
BOT = os.environ.get("BOT_URL", "http://localhost:8080")

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
RED, GREEN, YELLOW = "\033[91m", "\033[92m", "\033[93m"

sys.path.insert(0, str(Path(__file__).parent))
from run_judge import NousProvider  # noqa: E402  (reuses our key/UA handling)


def main() -> int:
    if not JUDGE.exists():
        print(f"judge_simulator.py not found at {JUDGE}")
        return 1
    if not KEY:
        print("Set LLM_API_KEY")
        return 1

    spec = importlib.util.spec_from_file_location("judge_simulator", JUDGE)
    js = importlib.util.module_from_spec(spec)
    sys.modules["judge_simulator"] = js
    spec.loader.exec_module(js)

    js.create_provider = lambda: NousProvider(KEY, JUDGE_MODEL)
    js.LLM_PROVIDER, js.LLM_API_KEY, js.LLM_MODEL = "nous", KEY, JUDGE_MODEL
    js.BOT_URL = BOT

    # The only change: ask for the reasoning. `_full` hardcodes verbose=False;
    # this substitutes a loop that reports the same scoring, with the reasons.
    original = js.JudgeSimulator._full
    rows: list[dict] = []

    def verbose_full(self) -> bool:
        """`_full`, but with the per-dimension rationales printed."""
        if not self._warmup():
            return False
        for mid, m in self.dataset.merchants.items():
            self.client.push_context("merchant", mid, 1, m)
        for tid, t in self.dataset.triggers.items():
            self.client.push_context("trigger", tid, 1, t)

        tids = list(self.dataset.triggers.keys())
        for i in range(0, len(tids), 5):
            batch = tids[i:i + 5]
            data, err, _lat = self.client.tick(batch)
            if err:
                continue
            for action in data.get("actions", []):
                self._score_and_display(action, verbose=True)
                rows.append({
                    "trigger_id": action.get("trigger_id", ""),
                    "body": action.get("body", ""),
                    "scores": self.all_scores[-1].total
                    if self.all_scores else None,
                })
        return True

    js.JudgeSimulator._full = verbose_full

    ok = js.JudgeSimulator(NousProvider(KEY, JUDGE_MODEL)).run("full_evaluation")

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"  scored {len(rows)} messages, reasons printed above")
    print(f"  bodies written to {Path('judge_verbose.json')}")
    Path("judge_verbose.json").write_text(json.dumps(rows, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
