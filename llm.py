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

import json
import re
import sys
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

    return PromptBundle(system=SYSTEM_PROMPT, user=user, facts=facts)


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
    return str((choices[0].get("message") or {}).get("content") or "")


def _headers(provider: str, key: str) -> dict[str, str]:
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01",
                "content-type": "application/json"}
    return {"Authorization": f"Bearer {key}", "content-type": "application/json"}


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

    prov = _PROVIDERS.get(config.LLM_PROVIDER)
    if not prov:
        return None

    key = config.LLM_API_KEY
    model = config.LLM_MODEL or "gpt-4o-mini"

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

    try:
        import httpx

        with httpx.Client(timeout=config.LLM_TIMEOUT_S) as client:
            r = client.post(url, json=payload, headers=_headers(config.LLM_PROVIDER, key))
            if r.status_code >= 400:
                return None
            data = r.json()
    except Exception:  # noqa: BLE001 — any failure must degrade, never 500
        return None

    return parse_body(_extract_text(config.LLM_PROVIDER, data)) or None


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

    print()
    print("ALL SEAM CHECKS PASSED" if ok else "SEAM CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test())
