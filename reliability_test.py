"""Provider reliability: bounded retry on 429, and visible degradation.

Two things are asserted here, and both are about things that are invisible by
default. A rate limit previously looked exactly like a good run — the bot
answered every trigger, never 500'd, and simply phrased everything
deterministically. And a retry that does not respect the tick deadline is
worse than no retry at all, because it trades a fine message for a timeout.
"""
import time
import types

from fastapi.testclient import TestClient

import llm
import server

_passed, _failed = [], []


def check(name, cond, detail=""):
    (_passed if cond else _failed).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" {detail}" if not cond and detail else ""))


def _resp(status, body=None, headers=None):
    r = types.SimpleNamespace()
    r.status_code = status
    r.headers = headers or {}
    r.json = lambda: (body if body is not None else {})
    return r


OK_BODY = {"choices": [{"message": {"content": "Hello from the provider."}}]}


class _FakeClient:
    """Stands in for httpx.Client; returns a scripted sequence of responses."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, *a, **k):
        self.calls += 1
        return self.script.pop(0) if self.script else _resp(429)


def _patch_script(monkey_script):
    import httpx

    holder = {}

    def factory(*a, **k):
        c = _FakeClient(monkey_script)
        holder["client"] = c
        return c

    httpx.Client = factory
    return holder


def main() -> int:
    import config

    print("Vera bot — provider reliability\n" + "=" * 60)

    # --- which statuses are worth another attempt -------------------------
    print("\n--- Retry policy ---")
    for s in (429, 500, 502, 503, 504):
        check(f"{s} is retryable", llm._retryable(s))
    for s in (400, 401, 403, 404, 422):
        check(f"{s} is not retryable", not llm._retryable(s))

    r = _resp(429, headers={"Retry-After": "30"})
    check("a huge Retry-After is clamped to the cap",
          llm._retry_delay(r, 1) <= llm.LLM_MAX_BACKOFF_S,
          f"got {llm._retry_delay(r, 1)}")
    check("a small Retry-After is honoured",
          abs(llm._retry_delay(_resp(429, headers={"Retry-After": "1"}), 1) - 1.0) < 1e-6)
    check("a date-form Retry-After does not crash and falls back to backoff",
          0 <= llm._retry_delay(_resp(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), 1)
          <= llm.LLM_MAX_BACKOFF_S)
    check("a missing Retry-Header yields exponential backoff",
          llm._retry_delay(_resp(429), 1) > llm._retry_delay(_resp(429), 0))
    check("backoff never returns a negative wait",
          llm._retry_delay(_resp(429, headers={"Retry-After": "-5"}), 1) >= 0.0)

    # --- the actual retry behaviour -------------------------------------
    print("\n--- Bounded retry against a scripted provider ---")
    # The provider is faked rather than configured: these assertions are about
    # our retry policy, and a test that needs a real key is a test that gets
    # skipped exactly when it matters most.
    import httpx

    real_client = httpx.Client
    real_prov, real_key = config.LLM_PROVIDER, config.LLM_API_KEY
    real_model = config.LLM_MODEL
    config.LLM_PROVIDER = "openai"
    config.LLM_API_KEY = "test-key-not-a-real-credential"
    config.LLM_MODEL = "test-model"

    def run(script, deadline=None):
        llm.reset_stats()
        # A successful call caches its body, and a cache hit skips the
        # provider entirely — so the cache must be cleared per scenario or
        # the second case never reaches the fake client at all.
        with llm._CACHE_LOCK:
            llm._CACHE.clear()
        b = llm.PromptBundle(system="s", user="u", model=config.LLM_MODEL)
        holder = _patch_script(script)
        try:
            out = llm.generate(b, deadline=deadline)
        finally:
            httpx.Client = real_client
        return out, holder["client"]

    try:
        out, c = run([_resp(429), _resp(200, OK_BODY)])
        check("a 429 followed by success returns the body",
              out == "Hello from the provider.", f"got {out!r}")
        check("the retry actually made a second call", c.calls == 2, f"calls={c.calls}")
        check("the retry is counted", llm.stats()["retries"] == 1)

        llm.reset_stats()
        out, c = run([_resp(429), _resp(429), _resp(429)])
        check("a persistently rate-limited provider gives up, it does not spin",
              out is None and c.calls <= llm.LLM_MAX_ATTEMPTS,
              f"calls={c.calls}, attempts allowed={llm.LLM_MAX_ATTEMPTS}")
        check("a rate-limited outcome is recorded as such",
              llm.stats()["rate_limited"] >= 1, str(llm.stats()))

        llm.reset_stats()
        out, c = run([_resp(401), _resp(200, OK_BODY)])
        check("a 401 is not retried", c.calls == 1, f"calls={c.calls}")
        check("a 401 still returns None rather than raising", out is None)

        # The deadline guard: no time left, so no retry, so no overrun.
        llm.reset_stats()
        out, c = run([_resp(429), _resp(200, OK_BODY)],
                     deadline=time.monotonic() + 0.05)
        check("a retry is skipped when the tick deadline is spent",
              c.calls == 1, f"calls={c.calls}")
        check("the skip is not counted as a retry", llm.stats()["retries"] == 0)

        t0 = time.monotonic()
        out, c = run([_resp(429), _resp(200, OK_BODY)], deadline=time.monotonic() + 30)
        check("with budget the retry does happen and is fast",
              c.calls == 2 and time.monotonic() - t0 < llm.LLM_MAX_BACKOFF_S + 3,
              f"calls={c.calls} elapsed={time.monotonic() - t0:.2f}s")

        llm.reset_stats()
        out, c = run([_resp(200, {"choices": [{"message": {"content": None}}]})])
        check("a 200 with an empty body is a fallback, not a crash", out is None)
    finally:
        httpx.Client = real_client
        config.LLM_PROVIDER, config.LLM_API_KEY = real_prov, real_key
        config.LLM_MODEL = real_model

    # --- observability ---------------------------------------------------
    print("\n--- Observability ---")
    s = llm.stats()
    for k in ("llm_ok", "fallback", "cached", "rate_limited", "retries"):
        check(f"stats expose {k}", k in s)

    client = TestClient(server.app)
    h = client.get("/v1/healthz").json()
    check("healthz still returns 200 and ok", h.get("status") == "ok")
    check("healthz reports composition counters", "composition" in h)
    check("healthz leaks no message content",
          all(k in {"llm_ok", "fallback", "cached", "rate_limited", "retries",
                    "provider_error", "timeout", "no_provider", "rejected"}
              for k in h.get("composition", {})),
          str(h.get("composition")))
    check("healthz still reports contexts_loaded", "contexts_loaded" in h)

    # The counter that catches a whole run going unphrased by the model.
    # Before this existed, an unconfigured provider produced zero counters
    # while the bot happily emitted 25 messages — the least honest possible
    # healthz: reporting nothing while quietly doing the thing you care about.
    import composer

    import signals as S

    sig = S.Signal(family="perf_dip", headline="calls 50% down", why_now="this week",
                   merchant_link="4 calls in 30d vs a 12 peer median")
    llm.reset_stats()
    real_prov, real_key = config.LLM_PROVIDER, config.LLM_API_KEY
    config.LLM_PROVIDER, config.LLM_API_KEY = "", ""
    try:
        composer._body_via_llm(sig, {}, {"merchant_id": "m_x"}, None, "2026-01-01T00:00:00Z")
    finally:
        config.LLM_PROVIDER, config.LLM_API_KEY = real_prov, real_key
    check("an unconfigured provider is counted, not silent",
          llm.stats()["no_provider"] == 1, str(llm.stats()))

    print("\n" + "=" * 60)
    print(f"  {len(_passed)}/{len(_passed) + len(_failed)} checks passed")
    for f in _failed:
        print(f"    x {f}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
