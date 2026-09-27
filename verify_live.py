#!/usr/bin/env python3
"""Verify all five judge endpoints against a deployed URL.

    BOT_URL=https://your-service.onrender.com .venv/bin/python verify_live.py

Timeouts are deliberately long. Render's free tier spins down after 15 minutes
idle and a cold start can take 30-50s, which is longer than the judge's own
30s budget — so a slow first call here is a cold start, not a broken endpoint.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("BOT_URL", "http://localhost:8080").rstrip("/")
DATASET = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")
COLD_START_BUDGET = 90

GREEN, RED, DIM, YELLOW, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[2m", "\033[93m", "\033[1m", "\033[0m"
)

results: list[tuple[str, bool, str]] = []


def req(method: str, path: str, body=None, timeout=COLD_START_BUDGET):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json", "User-Agent": "vera-verify/1.0"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read().decode()
            dt = time.perf_counter() - t0
            try:
                return resp.status, json.loads(raw), dt
            except json.JSONDecodeError:
                return resp.status, {"_raw": raw[:200]}, dt
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        dt = time.perf_counter() - t0
        try:
            return e.code, json.loads(raw or "{}"), dt
        except json.JSONDecodeError:
            return e.code, {"_raw": raw[:200]}, dt
    except Exception as e:  # noqa: BLE001
        return None, {"_err": f"{type(e).__name__}: {e}"}, time.perf_counter() - t0


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {name}{DIM}  {detail}{RESET}")


def main() -> int:
    print(f"{BOLD}{'=' * 74}{RESET}")
    print(f"  {BOLD}Live verification{RESET}  {BASE}")
    print(f"{BOLD}{'=' * 74}{RESET}\n")

    # 1. healthz
    st, d, dt = req("GET", "/v1/healthz", timeout=COLD_START_BUDGET)
    cold = " (cold start)" if dt > 10 else ""
    check("GET /v1/healthz", st == 200 and d.get("status") == "ok",
          f"{st} in {dt:.1f}s{cold}  contexts={d.get('contexts_loaded')}")
    loaded = d.get("contexts_loaded") or {}
    total = sum(loaded.values()) if isinstance(loaded, dict) else 0
    if dt > 30:
        print(f"      {YELLOW}NOTE: {dt:.0f}s response. Free-tier cold start. "
              f"Keep the service warm during judging.{RESET}")

    # 2. metadata
    st, d, dt = req("GET", "/v1/metadata")
    ok = st == 200 and d.get("team_name") and d.get("contact_email")
    check("GET /v1/metadata", ok, f"{st} in {dt:.1f}s  team={d.get('team_name')!r}")
    check("  team_members present", bool(d.get("team_members")),
          str(d.get("team_members")))
    check("  model reported", bool(d.get("model")), str(d.get("model")))

    # 3. context
    cat = json.loads(next((DATASET / "categories").glob("*.json")).read_text())
    st, d, _ = req("POST", "/v1/context", {
        "scope": "category", "context_id": cat["slug"], "version": 1, "payload": cat})
    check("POST /v1/context", st == 200 and d.get("accepted") is True, f"{st} {d}")
    st2, d2, _ = req("POST", "/v1/context", {
        "scope": "category", "context_id": cat["slug"], "version": 1, "payload": cat})
    check("  re-push same version is idempotent",
          st2 == 200 and d2.get("accepted") is True, f"{st2} {d2}")

    # 4. tick — seed one merchant + trigger so there is something to compose.
    merchants = json.loads((DATASET / "merchants_seed.json").read_text())["merchants"]
    triggers = json.loads((DATASET / "triggers_seed.json").read_text())["triggers"]
    for f in (DATASET / "categories").glob("*.json"):
        c = json.loads(f.read_text())
        req("POST", "/v1/context", {"scope": "category", "context_id": c["slug"],
                                    "version": 1, "payload": c})
    for m in merchants[:3]:
        req("POST", "/v1/context", {"scope": "merchant", "context_id": m["merchant_id"],
                                    "version": 1, "payload": m})
    picked = []
    seen = set()
    for t in triggers:
        if t["merchant_id"] not in seen:
            req("POST", "/v1/context", {"scope": "trigger", "context_id": t["id"],
                                        "version": 1, "payload": t})
            picked.append(t["id"])
            seen.add(t["merchant_id"])
        if len(picked) == 3:
            break

    st, d, dt = req("POST", "/v1/tick", {"available_triggers": picked}, timeout=40)
    acts = d.get("actions") or []
    check("POST /v1/tick", st == 200 and len(acts) > 0,
          f"{st} in {dt:.1f}s  {len(acts)}/{len(picked)} actions")
    check("  every action has a body", all(a.get("body") for a in acts))
    check("  no URL leaked into any body",
          not any("http://" in a["body"] or "https://" in a["body"] for a in acts))
    check("  no None/'None' text in bodies",
          not any("None" in a["body"] for a in acts))
    check("  within the judge's 30s budget", dt < 30, f"{dt:.1f}s")
    for a in acts:
        print(f"      {DIM}{a['body'][:88]}{RESET}")

    # 5. reply
    if acts:
        conv = acts[0]["conversation_id"]
        st, d, _ = req("POST", "/v1/reply", {
            "conversation_id": conv, "merchant_id": acts[0]["merchant_id"],
            "merchant_message": "Not interested, stop.", "turn_number": 1}, timeout=40)
        check("POST /v1/reply", st == 200 and d.get("action") in ("send", "wait", "end"),
              f"{st} action={d.get('action')!r}")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{BOLD}{'=' * 74}{RESET}")
    colour = GREEN if passed == len(results) else RED
    print(f"  {colour}{passed}/{len(results)} checks passed{RESET}")
    print(f"{BOLD}{'=' * 74}{RESET}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
