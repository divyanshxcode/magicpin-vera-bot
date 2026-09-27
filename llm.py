"""The LLM seam.

This is the ONE place the bot calls a model. Everything upstream — which
signal leads, which numbers support it, which category voice applies — is
decided in `signals.py` and `grounding.py` and arrives here already settled as
a `Signal`. This module's only job is to turn that settled decision into a
sentence that reads like a competent human wrote it.

The split is deliberate, and it is the whole architecture:

    signals.py + grounding.py  ->  WHAT to say, and the proof (code, audited)
    llm.py                     ->  HOW to phrase it          (model)

If a model were asked to also choose the signal, it would eventually invent a
figure that was never in the payload. The judge's anti-hallucination scoring
punishes exactly that, so the facts stay in code where they cannot drift.

PROVIDER SUPPORT
    Set LLM_PROVIDER to one of the keys in `_PROVIDERS`. The request is
    OpenAI-chat-shaped; Anthropic and Gemini adapters translate.
    With none set, `generate()` returns None and `composer.py` falls back to
    its deterministic implementation. Every endpoint keeps working either way,
    which is why the whole test suite runs with no API key.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

import config
import grounding as g
from signals import Signal

# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Vera, magicpin's merchant growth assistant. You write ONE WhatsApp \
message to an Indian small-business owner.

You are given the message DECISION already made by deterministic code. Your job \
is ONLY the wording. Never add a fact, number, date, price, offer, statistic or \
citation that is not present in the FACTS block. If a fact is missing, write \
around it — never fill the gap.

How to write:
- Lead with the single most useful fact. One idea per message.
- Be specific and concrete. Real numbers from the FACTS block, not adjectives.
- Match the CATEGORY VOICE described. It is not a suggestion.
- 2-4 short sentences. Under 320 characters. No links, ever.
- One clear next action, phrased so replying is a single word or tap.
- No emoji spam, no exclamation marks, no "Dear Sir/Madam", no corporate \
formality. You are a colleague, not a bank.
- Never reuse a phrasing the ALREADY SENT list contains.

Output ONLY the message body as plain text. No preamble, no quotes, no \
explanation, no signature."""


@dataclass
class PromptBundle:
    """Everything the model is told, and nothing it isn't."""

    system: str
    user: str
    facts: dict[str, Any] = field(default_factory=dict)
    #: Sampling settings this bundle was built for. Bound at construction
    #: rather than read from config later, so the cache key cannot shift
    #: underneath a request if config changes mid-flight (the tick path runs
    #: composition from several threads).
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    #: "none" | "low" | "" — disables hybrid-model reasoning where supported.
    reasoning_effort: str = ""


def _voice_block(category: dict) -> str:
    v = category.get("voice") or {}
    bits = []
    for key in ("register", "tone", "avoid", "taboo_words", "do_not_say"):
        val = v.get(key)
        if not val:
            continue
        if isinstance(val, (list, tuple)):
            val = ", ".join(str(x) for x in val)
        bits.append(f"  {key}: {val}")
    return "CATEGORY VOICE\n" + "\n".join(bits) if bits else ""


def _fact_line(label: str, value: Any) -> str:
    text = str(value or "").strip()
    return f"  {label}: {text}" if text else ""


def build_prompt(
    signal: Signal,
    category: dict,
    merchant: dict,
    customer: Optional[dict] = None,
    already_sent: Optional[list[str]] = None,
) -> PromptBundle:
    """Assemble the prompt from a decided Signal.

    Note what is NOT here: no merchant name guessing, no date arithmetic, no
    slot selection. Those were already resolved upstream, so the model cannot
    get them wrong — it can only phrase them badly, which is recoverable.
    """
    facts: dict[str, Any] = {
        "the_point": signal.headline or signal.question or "",
        "why_now": signal.why_now,
        "supporting_fact": signal.merchant_link,
        "citation": signal.citation,
        "specific_offer": (g.offer_title(signal.offer) if signal.offer else ""),
        "offer_price": (g.offer_price_text(signal.offer) if signal.offer else ""),
        "slots": [s.get("label", "") for s in (signal.slots or []) if s.get("label")],
        "cta_to_ask_for": signal.cta,
        "audience": ("a customer of this business" if signal.customer_facing
                     else "the business owner"),
    }

    if customer:
        facts["recipient_first_name"] = g.customer_first_name(customer)
        facts["recipient_state"] = g.dig(customer, "state", default="")
        facts["language_preference"] = g.dig(
            customer, "identity", "language_pref", default="")

    lines = [f"  {k}: {v}" for k, v in facts.items() if v not in ("", None, [])]
    if already_sent:
        lines.append("")
        lines.append("ALREADY SENT (do not repeat these phrasings):")
        lines.extend(f"  - {b}" for b in already_sent[-3:])

    user = "\n".join([
        "FACTS (use these and only these):",
        "\n".join(lines),
        "",
        _voice_block(category),
        "",
        f"Write the {facts['audience']}'s message now. Body text only.",
    ]).strip()

    return PromptBundle(system=SYSTEM_PROMPT, user=user, facts=facts,
                        model=config.LLM_MODEL,
                        temperature=config.LLM_TEMPERATURE,
                        max_tokens=config.LLM_MAX_TOKENS,
                        reasoning_effort=config.REASONING_EFFORT)


# --------------------------------------------------------------------------
# response parsing
# --------------------------------------------------------------------------

_TRAILING_JUNK = re.compile(r"^[\"'`]+|[\"'`]+$")
_URL = re.compile(r"https?://\S+|www\.\S+")
# Prefixes models prepend to the real body. Stripped wherever they appear,
# single-line included — "Message: Your listing is live" is one line, so a
# line-based filter alone would let it through.
_PREFIX_RE = re.compile(
    r"^\s*(?:here'?s\s+(?:is\s+)?(?:the\s+|a\s+)?(?:message|body|reply|draft)"
    r"|message|body|reply|draft|output|answer)\s*[:\-–]\s*",
    re.IGNORECASE,
)

# Cloudflare fronts some inference endpoints and rejects the default
# httpx/python-httpx user-agent with an HTML 403 WAF page, which parse_body
# would then read as a model response. Send an ordinary one.
_DEFAULT_UA = "magicpin-vera-bot/0.1"


def parse_body(raw: str) -> str:
    """Pull a clean body out of whatever the model returned.

    Models wrap answers in quotes, add "Message:" prefixes, or emit a
    reasoning preamble. Strip all of it — the validator downstream is strict
    and we would rather hand it something plausible than let it reject the
    whole send.
    """
    text = (raw or "").strip()
    if not text:
        return ""

    if "\n" in text:
        # Drop whole preamble lines, then rejoin the survivors.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        lines = [ln for ln in lines if not _PREFIX_RE.match(ln)]
        text = " ".join(lines) if len(lines) > 1 else (lines[0] if lines else "")

    # Same prefix, inline.
    text = _PREFIX_RE.sub("", text.strip())

    text = _TRAILING_JUNK.sub("", text.strip())
    text = _URL.sub("", text)          # belt and braces; validate.py also does this
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# --------------------------------------------------------------------------
# provider adapters
# --------------------------------------------------------------------------


@dataclass
class _Provider:
    url: str
    key_env: str = "LLM_API_KEY"
    style: str = "openai"


_PROVIDERS: dict[str, _Provider] = {
    "openai": _Provider("https://api.openai.com/v1/chat/completions"),
    "deepseek": _Provider("https://api.deepseek.com/chat/completions"),
    "groq": _Provider("https://api.groq.com/openai/v1/chat/completions"),
    "openrouter": _Provider("https://openrouter.ai/api/v1/chat/completions"),
    "nous": _Provider("https://inference-api.nousresearch.com/v1/chat/completions"),
    "anthropic": _Provider("https://api.anthropic.com/v1/messages", style="anthropic"),
    "gemini": _Provider(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent?key={key}", style="gemini"),
}


def _to_anthropic(bundle: PromptBundle, model: str) -> dict:
    return {
        "model": model,
        "max_tokens": config.LLM_MAX_TOKENS,
        "temperature": config.LLM_TEMPERATURE,
        "system": bundle.system,
        "messages": [{"role": "user", "content": bundle.user}],
    }


def _to_gemini(bundle: PromptBundle, model: str) -> dict:
    return {
        "contents": [{"parts": [{"text": bundle.user}]}],
        "systemInstruction": {"parts": [{"text": bundle.system}]},
        "generationConfig": {
            "maxOutputTokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
        },
    }


def _extract_text(provider: str, data: dict) -> str:
    if provider == "anthropic":
        parts = [b.get("text", "") for b in data.get("content", [])
                 if isinstance(b, dict)]
        return "".join(parts)
    if provider == "gemini":
        cands = data.get("candidates") or []
        if not cands:
            return ""
        return "".join(p.get("text", "")
                       for p in cands[0].get("content", {}).get("parts", []))
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    # A hybrid reasoning model that exhausts its budget mid-thought returns
    # content=None with finish_reason="length". There is no message to ship, so
    # return "" and let the deterministic composer cover it.
    content = msg.get("content")
    if content:
        return str(content)
    # Gemini sometimes puts the answer under a different key; some reasoning
    # models leave it in the refusal field.
    return str(msg.get("refusal") or "")


def _headers(provider: str, key: str) -> dict[str, str]:
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01",
                "content-type": "application/json", "user-agent": _DEFAULT_UA}
    return {"Authorization": f"Bearer {key}", "content-type": "application/json",
            "accept": "application/json", "user-agent": _DEFAULT_UA}


# --------------------------------------------------------------------------
# output cache
# --------------------------------------------------------------------------
# The judge requires "your output should stay deterministic for the same input
# and simulator settings". Measured against this provider, temperature 0 is
# NOT honoured — the same prompt returns different wording across calls, on
# both stealth/space-bunny-alpha and poolside/laguna-s-2.1:free. The prompt
# bytes hash identically; the sampling does not.
#
# So determinism has to come from us. Keying on the full prompt hash makes
# composition a pure function of the injected context: the same trigger always
# yields the same message for the life of the process. Replayed ticks, a second
# judge pass, and retried requests all get identical text.
#
# A replayed trigger is normally suppressed anyway, so this mainly matters for
# judge retries and for the same signal resurfacing on a later tick.

_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()


def _cache_key(bundle: PromptBundle) -> str:
    raw = "\x1f".join([
        bundle.model,
        str(bundle.temperature),
        str(bundle.max_tokens),
        str(bundle.reasoning_effort),
        bundle.system,
        bundle.user,
    ])
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_get(bundle: PromptBundle) -> Optional[str]:
    with _CACHE_LOCK:
        return _CACHE.get(_cache_key(bundle))


def cache_put(bundle: PromptBundle, body: str) -> None:
    with _CACHE_LOCK:
        _CACHE[_cache_key(bundle)] = body


def cache_size() -> int:
    with _CACHE_LOCK:
        return len(_CACHE)


def _cache_get_is_miss(bundle: PromptBundle) -> bool:
    """Test helper: True when nothing is stored for this bundle."""
    return cache_get(bundle) is None


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------


def generate(bundle: PromptBundle) -> Optional[str]:
    """Call the configured model. Returns None if unavailable or on any failure.

    Returning None rather than raising is load-bearing: the judge treats a 500
    as a hard failure, so a provider outage, a rate limit or a malformed
    response must degrade to the deterministic composer, never to an error.
    """
    if not config.llm_configured():
        return None

    cached = cache_get(bundle)
    if cached is not None:
        return cached

    prov = _PROVIDERS.get(config.LLM_PROVIDER)
    if not prov:
        return None

    key = config.LLM_API_KEY
    # Use the bundle's bound model, not a fresh read of config — the bundle is
    # what the cache was keyed on, so the request must match it exactly.
    model = bundle.model or config.LLM_MODEL or "gpt-4o-mini"

    if prov.style == "anthropic":
        url, payload = prov.url, _to_anthropic(bundle, model)
    elif prov.style == "gemini":
        url = prov.url.format(model=model, key=key)
        payload = _to_gemini(bundle, model)
    else:
        url, payload = prov.url, {
            "model": model,
            "messages": [{"role": "system", "content": bundle.system},
                         {"role": "user", "content": bundle.user}],
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
        }
        # Hybrid reasoning models accept this; plain ones ignore unknown keys,
        # so it is safe to send whenever it is configured.
        if bundle.reasoning_effort:
            payload["reasoning_effort"] = bundle.reasoning_effort

    try:
        import httpx

        with httpx.Client(timeout=config.LLM_TIMEOUT_S) as client:
            r = client.post(url, json=payload, headers=_headers(config.LLM_PROVIDER, key))
            if r.status_code >= 400:
                return None
            data = r.json()
    except Exception:  # noqa: BLE001 — any failure must degrade, never 500
        return None

    body = parse_body(_extract_text(config.LLM_PROVIDER, data)) or None
    if body:
        # Only successful, non-empty responses are cached — a transient failure
        # must not pin the bot to a fallback body for the life of the process.
        cache_put(bundle, body)
    return body


# --------------------------------------------------------------------------
# output cache
# --------------------------------------------------------------------------


def compose_many(
    jobs: list[tuple],
    deadline: Optional[float] = None,
) -> list[Optional[str]]:
    """Phrase several signals at once, concurrently.

    `jobs` is a list of (category, merchant, trigger, customer, now) tuples.
    Sequential calls measured 1.6-7.0s each against this provider, so a tick
    with several actions would blow the judge's 30s budget. Threads are the
    right tool here: the work is pure network wait, and httpx releases the GIL
    while blocked.

    `deadline` is a time.monotonic() value. Any job still outstanding when it
    passes returns None, which routes that message to the deterministic
    composer — so a slow provider degrades the prose, never the availability.
    """
    if not jobs:
        return []
    if not config.llm_configured() or len(jobs) == 1:
        return [_compose_one(*j) for j in jobs]

    results: list[Optional[str]] = [None] * len(jobs)

    def work(idx: int, job: tuple) -> None:
        results[idx] = _compose_one(*job)

    with ThreadPoolExecutor(max_workers=min(len(jobs), 8)) as pool:
        futures = {pool.submit(work, i, j): i for i, j in enumerate(jobs)}
        for fut in as_completed(futures, timeout=_remaining(deadline)):
            idx = futures[fut]
            try:
                fut.result()
            except Exception:  # noqa: BLE001
                results[idx] = None

    return results


def _remaining(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def _compose_one(category: dict, merchant: dict, trigger: dict,
                 customer: Optional[dict], now: str,
                 sent_bodies: Optional[set] = None) -> Optional[str]:
    """Full compose() for one job, used by the parallel path."""
    from composer import compose as _compose

    out = _compose(category, merchant, trigger, customer, now, sent_bodies)
    return out["body"] if out else None


def self_test() -> int:
    """Prove the seam is wired, without needing a key.

    Verifies prompt construction, the no-invention contract, and that an
    unconfigured provider returns None rather than raising.
    """
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  {'[PASS]' if cond else '[FAIL]'} {name}"
              + (f"  {detail}" if detail and not cond else ""))

    cat = {"slug": "dentists", "voice": {
        "register": "clinical, precise, peer-to-peer",
        "avoid": "salesy hype",
        "taboo_words": ["cure", "guaranteed", "painless"]}}
    merchant = {"merchant_id": "m1", "identity": {"name": "Dr. Meera's Dental Clinic"},
                "category_slug": "dentists"}
    sig = Signal(family="research", headline="3-month fluoride varnish recall "
                 "outperforms 6-month (2,100-patient trial)",
                 why_now="this week's research just landed",
                 citation="JIDA Oct 2026, p.14",
                 merchant_link="124 high-risk adult patients on your list",
                 cta="binary_yes_no", lever="authority")

    bundle = build_prompt(sig, cat, merchant)

    print("--- prompt construction ---")
    check("system prompt forbids invented facts",
          "not present in the FACTS block" in bundle.system)
    check("system prompt forbids links", "no links" in bundle.system.lower())
    check("system prompt demands body-only output",
          "Output ONLY the message body" in bundle.system)
    check("headline reaches the prompt", "fluoride" in bundle.user)
    check("citation reaches the prompt", "JIDA Oct 2026, p.14" in bundle.user)
    check("merchant link reaches the prompt", "124 high-risk" in bundle.user)
    check("category voice reaches the prompt", "clinical, precise" in bundle.user)
    check("taboo words reach the prompt", "guaranteed" in bundle.user)
    check("CTA reaches the prompt", "binary_yes_no" in bundle.user)

    print("--- no-invention contract ---")
    # The prompt must never hand the model a number that is not in the Signal.
    prompt_numbers = set(re.findall(r"\d[\d,.]*", bundle.user))
    known = set(re.findall(r"\d[\d,.]*",
                            " ".join(str(v) for v in bundle.facts.values())))
    check("every number in the prompt traces to a fact",
          prompt_numbers <= known,
          f"unexplained: {prompt_numbers - known}")

    print("--- response parsing ---")
    for raw, expect in [
        ('"Hello there, quick one."', "Hello there, quick one."),
        ("Message: Your listing is live", "Your listing is live"),
        ("Here's the message:\n\nYour post is live", "Your post is live"),
        ("Visit https://x.com now", "Visit now"),
        ("  spaced   out  ", "spaced out"),
        ("", ""),
    ]:
        got = parse_body(raw)
        check(f"parse {raw[:28]!r}", got == expect, f"got {got!r}")

    print("--- unconfigured path degrades cleanly ---")
    if config.llm_configured():
        check("provider is configured; skipping None-return check", True)
    else:
        check("generate() returns None with no API key", generate(bundle) is None)

    print("--- build_prompt must not raise for any signal shape ---")
    # This is the check that would have caught the offer_price_text bug: a
    # customer-facing signal carrying an offer made build_prompt raise, and the
    # try/except in composer._body_via_llm swallowed it into a silent
    # deterministic fallback. Every branch of build_prompt needs to be built.
    import grounding as gg

    merchant_offer = {"merchant_id": "m1",
                      "identity": {"name": "Dr. Meera's Dental Clinic"},
                      "category_slug": "dentists",
                      "offers": [{"id": "o1", "title": "Dental Cleaning @ ₹299",
                                  "price": 299, "original_price": 499,
                                  "status": "active", "valid_until": "2026-06-30"}]}
    shapes = [
        ("plain research", Signal(family="research", headline="H",
                                  why_now="w", citation="c", merchant_link="l",
                                  cta="binary_yes_no", lever="authority")),
        ("offer only", Signal(family="offer", headline="H", why_now="w",
                              merchant_link="l", cta="open_ended",
                              offer=gg.active_offers(merchant_offer, limit=1)[0])),
        ("with slots", Signal(family="offer", headline="H", why_now="w",
                              merchant_link="l", cta="multi_choice_slot",
                              slots=[{"iso": "2026-05-01T10:00:00+05:30",
                                      "label": "Fri 1 May, 10am"}])),
        ("question only", Signal(family="curious", headline="",
                                  question="What's your best "
                                  "seller this month?", why_now="n",
                                  cta="open_ended", lever="curiosity")),
        ("customer facing", Signal(family="recall", headline="H", why_now="w",
                                   merchant_link="l", cta="multi_choice_slot",
                                   customer_facing=True, slots=[{"label": "x"}],
                                   offer=gg.active_offers(merchant_offer, limit=1)[0])),
    ]
    customer_ctx = {"customer_id": "c1",
                    "identity": {"name": "Priya", "language_pref": "hi-en mix"},
                    "state": "lapsed_soft"}
    for name, s in shapes:
        try:
            pb = build_prompt(s, cat, merchant_offer,
                              customer_ctx if s.customer_facing else None)
            okb = bool(pb.user) and bool(pb.system)
        except Exception as e:  # noqa: BLE001
            okb, pb = False, None
            print(f"        {DIM}{type(e).__name__}: {e}{RESET}")
        check(f"build_prompt survives: {name}", okb)

    print("--- offer price rendering ---")
    check("plain price", gg.offer_price_text({"price": 299}) == "₹299",
          gg.offer_price_text({"price": 299}))
    check("discount shows the was-price",
          gg.offer_price_text({"price": 299, "original_price": 499}) == "₹299 (was ₹499)",
          gg.offer_price_text({"price": 299, "original_price": 499}))
    check("no price -> empty string", gg.offer_price_text({"title": "x"}) == "")
    check("zero/negative price -> empty string",
          gg.offer_price_text({"price": 0}) == "" and gg.offer_price_text({"price": -5}) == "")
    check("None offer -> empty string", gg.offer_price_text(None) == "")
    check("was-price lower than price is not shown",
          gg.offer_price_text({"price": 500, "original_price": 100}) == "₹500",
          gg.offer_price_text({"price": 500, "original_price": 100}))

    print("--- output cache (determinism) ---")
    # temperature 0 is not honoured by this provider, so identical prompts can
    # return different bodies. The cache is what makes composition a pure
    # function of the injected context, which is what the judge requires.
    import config as _c
    was_prov, was_key, was_model = _c.LLM_PROVIDER, _c.LLM_API_KEY, _c.LLM_MODEL
    try:
        _c.LLM_PROVIDER, _c.LLM_API_KEY, _c.LLM_MODEL = "openai", "k", "m1"
        b1 = build_prompt(sig, cat, merchant)
        b2 = build_prompt(sig, cat, merchant)
        check("identical signals produce identical prompts",
              b1.user == b2.user and b1.system == b2.system)
        check("cache key is stable", _cache_key(b1) == _cache_key(b2))

        cache_put(b1, "FIRST BODY")
        check("cache returns what was stored", cache_get(b1) == "FIRST BODY")
        check("second read is identical", cache_get(b2) == "FIRST BODY")

        # An already-built bundle must keep its key when config changes later.
        # (This is the whole reason the model is bound onto the bundle rather
        # than read from config at cache time.)
        key_before = _cache_key(b1)
        _c.LLM_MODEL = "m2"
        check("an already-built bundle keeps its key",
              _cache_key(b1) == key_before,
              "b1's key shifted when LLM_MODEL changed")

        # A bundle built NOW must pick up the new model.
        b4 = build_prompt(sig, cat, merchant)
        check("a newly built bundle binds the new model",
              b4.model == "m2" and _cache_get_is_miss(b4),
              f"b4.model={b4.model!r}")

        # And a different prompt must not collide.
        b3 = build_prompt(sig, cat, merchant, already_sent=["something else"])
        check("different prompt -> different key", _cache_get_is_miss(b3))

        before = cache_size()
        cache_put(b3, "SECOND BODY")
        check("cache grows with distinct prompts", cache_size() == before + 1,
              f"{before} -> {cache_size()}")
    finally:
        _c.LLM_PROVIDER, _c.LLM_API_KEY, _c.LLM_MODEL = was_prov, was_key, was_model

    print("--- hybrid reasoning models ---")
    # A reasoning model that exhausts its budget mid-thought returns
    # content=None with finish_reason="length". Shipping that as a body would
    # put the string "None" in front of a merchant, so it must extract to "".
    check("null content extracts to empty (not the string 'None')",
          _extract_text("openai", {"choices": [{"message": {"content": None},
                                                "finish_reason": "length"}]}) == "",
          repr(_extract_text("openai", {"choices": [{"message": {"content": None}}]})))
    check("normal content still extracts",
          _extract_text("openai", {"choices": [{"message": {"content": "hi"}}]}) == "hi")
    check("missing choices is empty",
          _extract_text("openai", {"choices": []}) == "")
    check("missing message key is empty",
          _extract_text("openai", {"choices": [{}]}) == "")
    check("null content survives parse_body as empty",
          parse_body(_extract_text("openai",
                                   {"choices": [{"message": {"content": None}}]})) == "")

    print("--- reasoning_effort plumbing ---")
    import config as _c
    was_effort = _c.REASONING_EFFORT
    try:
        _c.REASONING_EFFORT = "none"
        b_eff = build_prompt(sig, cat, merchant)
        check("bundle binds reasoning_effort", b_eff.reasoning_effort == "none",
              repr(b_eff.reasoning_effort))
        check("reasoning_effort is part of the cache key",
              _cache_key(b_eff) != _cache_key(b1))
        _c.REASONING_EFFORT = ""
        b_noeff = build_prompt(sig, cat, merchant)
        check("blank effort produces a different key than 'none'",
              _cache_key(b_noeff) != _cache_key(b_eff))
    finally:
        _c.REASONING_EFFORT = was_effort

    print()
    print("ALL SEAM CHECKS PASSED" if ok else "SEAM CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test())
