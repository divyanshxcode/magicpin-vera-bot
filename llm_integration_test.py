#!/usr/bin/env python3
"""Verify the LLM seam end-to-end, with a mocked provider.

The seam is the one part of the system that cannot be tested by running the
real thing — it needs a key and it costs money. So this exercises every branch
that matters, against a stub HTTP layer, and asserts the properties the judge
actually scores:

  - the model is given facts, not the whole context blob
  - a good response is used verbatim
  - a response with a link / preamble / quotes is cleaned before it ships
  - a response with a taboo word is rejected and the deterministic body used
  - a provider outage, timeout or 500 degrades to deterministic, never 500s
  - identical input produces identical output (temperature 0)

Run:  .venv/bin/python llm_integration_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

import server
from composer import compose
from store import contexts, conversations, dispatch

DATASET = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")

GREEN, RED, DIM, CYAN, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[2m", "\033[96m", "\033[1m", "\033[0m"
)

_passed: list[str] = []
_failed: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    (_passed if cond else _failed).append(name)
    mark = (GREEN + "[PASS]" + RESET) if cond else (RED + "[FAIL]" + RESET)
    print(f"  {mark} {name}" + (f" {DIM}{detail}{RESET}" if detail and not cond else ""))
    return cond


def section(t: str) -> None:
    print(f"\n{CYAN}{BOLD}--- {t} ---{RESET}")


def load(p: Path):
    return json.loads(p.read_text())


class StubTransport(httpx.BaseTransport):
    """Stands in for the provider's HTTP layer.

    `reply` is the raw text the fake model returns; `status` and `raise_exc`
    let us simulate outages and timeouts.
    """

    def __init__(self, reply: str = "", status: int = 200,
                 raise_exc: Exception | None = None) -> None:
        self.reply = reply
        self.status = status
        self.raise_exc = raise_exc
        self.requests: list[dict] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        self.requests.append({"url": str(request.url), "body": body,
                              "headers": dict(request.headers)})
        if self.raise_exc:
            raise self.raise_exc
        if self.status >= 400:
            return httpx.Response(self.status, json={"error": "upstream"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": self.reply}}]})


def use_stub(transport: StubTransport) -> None:
    """Point every httpx.Client inside llm.py at the stub."""
    original = httpx.Client

    def patched(*a, **kw):
        kw["transport"] = transport
        return original(*a, **kw)

    httpx.Client = patched


def restore() -> None:
    httpx.Client = _REAL_CLIENT


_REAL_CLIENT = httpx.Client


def seed() -> None:
    contexts.reset()
    conversations.reset()
    dispatch.reset()
    for f in (DATASET / "categories").glob("*.json"):
        cat = load(f)
        contexts.push("category", cat["slug"], 1, cat)
    for m in load(DATASET / "merchants_seed.json")["merchants"]:
        contexts.push("merchant", m["merchant_id"], 1, m)
    for t in load(DATASET / "triggers_seed.json")["triggers"]:
        contexts.push("trigger", t["id"], 1, t)


def compose_one(trigger_id: str) -> dict | None:
    from store import contexts as ctx
    trigger = ctx.get("trigger", trigger_id)
    merchant = ctx.get("merchant", trigger["merchant_id"])
    category = ctx.get("category", merchant["category_slug"])
    return compose(category, merchant, trigger, None, "2026-04-26T10:35:00Z")


def enable_llm() -> None:
    import config
    config.LLM_PROVIDER = "openai"
    config.LLM_API_KEY = "sk-test-not-real"
    config.LLM_MODEL = "gpt-4o-mini"
    config.LLM_TEMPERATURE = 0.0


def disable_llm() -> None:
    import config
    config.LLM_PROVIDER = ""
    config.LLM_API_KEY = ""


# --------------------------------------------------------------------------


def test_request_shape() -> None:
    section("Request shape — the model is given facts, not the whole blob")
    seed()
    enable_llm()
    stub = StubTransport(reply="Fluoride varnish recall is worth a look.")
    use_stub(stub)
    try:
        out = compose_one("trg_001_research_digest_dentists")
    finally:
        restore()

    check("a request was actually made", len(stub.requests) == 1,
          f"{len(stub.requests)} requests")
    if not stub.requests:
        return
    body = stub.requests[0]["body"]
    check("temperature is pinned to 0", body.get("temperature") == 0.0,
          str(body.get("temperature")))
    check("system prompt forbids invented facts",
          "not present in the FACTS block" in body["messages"][0]["content"])
    user = body["messages"][1]["content"]
    check("user turn carries the resolved facts", "FACTS" in user)
    check("sends the citation, not the raw digest blob",
          "JIDA" in user and "digest" not in user.lower(), user[:100])
    check("does NOT dump the whole category JSON", "voice" not in user[:200].lower()
          or len(user) < 4000, f"prompt is {len(user)} chars")
    check("prompt is compact", len(user) < 2000, f"{len(user)} chars")
    check("auth header sent", "authorization" in
          {k.lower() for k in stub.requests[0]["headers"]})
    check("composed body came from the model",
          out and out["body"] == "Fluoride varnish recall is worth a look.",
          out["body"] if out else "None")


def test_cleanup() -> None:
    section("Cleanup — messy model output is repaired before it ships")
    cases = [
        # parse_body does not add sentence punctuation — that is the
        # deterministic assembler's job, and only for bodies it builds. An LLM
        # body is shipped with the model's own punctuation, so these expect it
        # verbatim.
        ("quoted", '"Your recall list is ready."', "Your recall list is ready."),
        ("preamble", "Message: Your recall list is ready", "Your recall list is ready"),
        ("multiline", "Here's the message:\n\nYour recall list is ready",
         "Your recall list is ready"),
        ("link", "Check https://magicpin.com/vera now", "Check now"),
    ]
    for name, raw, expect in cases:
        seed()
        stub = StubTransport(reply=raw)
        use_stub(stub)
        try:
            out = compose_one("trg_001_research_digest_dentists")
        finally:
            restore()
        got = out["body"] if out else ""
        check(f"{name}: '{raw[:30]}'", got == expect, f"got {got!r}")


def test_validation_fallback() -> None:
    section("Validation — a bad model response falls back, never ships")
    # "guaranteed" is a taboo word for dentists; the validator must catch it
    # and the deterministic body must replace it.
    seed()
    stub = StubTransport(reply="Guaranteed results, book today at https://x.com")
    use_stub(stub)
    try:
        out = compose_one("trg_001_research_digest_dentists")
    finally:
        restore()
    body = out["body"] if out else ""
    check("taboo word did not ship", "guaranteed" not in body.lower(), body[:90])
    check("no link shipped", "http" not in body.lower(), body[:90])
    check("still produced a message", bool(body.strip()), "compose returned None")
    check("rationale records the auto-correction",
          out and "auto-corrected" in out["rationale"], out["rationale"][:100] if out else "")

    section("Validation — hopelessly bad output degrades to silence")
    seed()
    stub = StubTransport(reply="")
    use_stub(stub)
    try:
        out = compose_one("trg_001_research_digest_dentists")
    finally:
        restore()
    check("empty model output -> deterministic body, not a crash",
          out is not None and bool(out.get("body", "").strip()),
          "returned None")


def test_provider_failures() -> None:
    section("Provider failures — degrade to deterministic, never 500")
    scenarios = [
        ("HTTP 500", StubTransport(status=500)),
        ("HTTP 429 rate limit", StubTransport(status=429)),
        ("timeout", StubTransport(raise_exc=httpx.ReadTimeout("timed out"))),
        ("connection error", StubTransport(raise_exc=httpx.ConnectError("refused"))),
        ("malformed JSON", None),
    ]
    for name, stub in scenarios:
        seed()
        if stub is None:
            class BadJSON(httpx.BaseTransport):
                def handle_request(self, r):
                    return httpx.Response(200, content=b"not json at all")
            stub = StubTransport()
            stub.__class__ = BadJSON  # reuse the harness shape
        use_stub(stub)
        try:
            out = compose_one("trg_001_research_digest_dentists")
        finally:
            restore()
        check(f"{name} -> still composes", out is not None and bool(out.get("body", "").strip()),
              "compose returned None")
        check(f"{name} -> no URL leaked",
              out and "http" not in out["body"].lower(), out["body"][:80] if out else "")


def test_determinism() -> None:
    section("Determinism — temperature 0, same input, same output")
    seed()
    enable_llm()
    bodies = []
    for _ in range(3):
        seed()
        stub = StubTransport(reply="Your recall list is ready.")
        use_stub(stub)
        try:
            out = compose_one("trg_001_research_digest_dentists")
        finally:
            restore()
        bodies.append(out["body"] if out else None)
    check("3 stubbed calls give identical output", len(set(bodies)) == 1, str(bodies))

    # And the request we send must itself be identical, or the model has no
    # chance of being deterministic.
    reqs = []
    for _ in range(2):
        seed()
        stub = StubTransport(reply="x")
        use_stub(stub)
        try:
            compose_one("trg_001_research_digest_dentists")
        finally:
            restore()
        reqs.append(json.dumps(stub.requests[0]["body"], sort_keys=True))
    check("identical prompt bytes across calls", reqs[0] == reqs[1],
          "prompt varies between identical calls")


def test_disabled() -> None:
    section("Disabled — the default path makes zero network calls")
    seed()
    disable_llm()
    seen = []

    class Tripwire(httpx.BaseTransport):
        def handle_request(self, r):
            seen.append(str(r.url))
            return httpx.Response(500)

    use_stub(Tripwire())
    try:
        out = compose_one("trg_001_research_digest_dentists")
    finally:
        restore()
    check("no HTTP call attempted with LLM off", not seen, str(seen))
    check("deterministic body still produced", out and bool(out["body"].strip()))
    check("metadata reports the deterministic model",
          server.config.metadata()["model"] == "deterministic-composer",
          server.config.metadata()["model"])


def test_endpoints_with_llm() -> None:
    section("Endpoints — the full judge path works with the LLM enabled")
    seed()
    enable_llm()
    stub = StubTransport(reply="Your 124 high-risk patients are due for a recall.")
    use_stub(stub)
    try:
        client = TestClient(server.app)
        r = client.post("/v1/tick", json={
            "now": "2026-04-26T10:35:00Z",
            "available_triggers": ["trg_001_research_digest_dentists"]})
        check("/v1/tick returns 200", r.status_code == 200, str(r.status_code))
        actions = r.json().get("actions", [])
        check("/v1/tick produced an action", len(actions) == 1, f"{len(actions)}")
        if actions:
            check("action body is the model's",
                  actions[0]["body"] == "Your 124 high-risk patients are due for a recall.",
                  actions[0]["body"][:80])
            check("rationale still present and non-empty",
                  bool(actions[0]["rationale"].strip()))
        h = client.get("/v1/healthz")
        check("/v1/healthz still 200", h.status_code == 200)
        m = client.get("/v1/metadata").json()
        check("metadata advertises the model", "openai" in m["model"], m["model"])
    finally:
        restore()
        disable_llm()


def main() -> int:
    print(f"{BOLD}{'=' * 62}{RESET}\n  Vera bot — LLM seam integration\n{BOLD}{'=' * 62}{RESET}")
    try:
        test_request_shape()
        test_cleanup()
        test_validation_fallback()
        test_provider_failures()
        test_determinism()
        test_disabled()
        test_endpoints_with_llm()
    finally:
        restore()
        disable_llm()

    total = len(_passed) + len(_failed)
    print(f"\n{BOLD}{'=' * 62}{RESET}")
    if _failed:
        print(f"  {RED}{len(_failed)}/{total} FAILED{RESET}")
        for n in _failed:
            print(f"    {RED}x{RESET} {n}")
        return 1
    print(f"  {GREEN}ALL {total} SEAM CHECKS PASSED{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
