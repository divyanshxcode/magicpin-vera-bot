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
- OPEN BY ADDRESSING THE RECIPIENT BY NAME, using `recipient_first_name`. \
Keep any title it carries ("Dr. Meera", not "Meera"). Then a comma, then the \
message. A message that opens with a fact instead of a name reads as a bulk \
broadcast and is the single fastest way to lose the reader.
- If `recipient_first_name` is ABSENT, the name is genuinely unknown. Do not \
invent one, and do not borrow a name from anywhere. Open with the business \
instead — "This is <their_business> —" — and let the rest of the message do the \
talking.
- When `audience` is a customer, the message is sent from the merchant's \
WhatsApp number, so attribute the business in the first sentence. A customer \
who cannot tell which business is writing has no reason to read it.
- If `question_to_ask` is present, the message must actually ASK it. It is the \
point of the message, not an optional flourish. Ask it, then make answering \
worth their time.
- Lead with the single most useful fact. One idea per message.
- Apply `compulsion_lever` with `how_to_apply_it`. That lever is the reason \
this message exists, not decoration — a loss-aversion message that reads like \
curiosity wastes the one thing that earns a reply. Vary it across \
conversations; a merchant who always gets the same shape stops opening them.
- Tie the fact to THIS merchant. Their own numbers, customers, offer or area \
beat an abstract statement about the market. "124 of your patients are in \
this trial" lands; "high-risk adults need more frequent recalls" does not.
- Be specific and concrete. Real numbers from the FACTS block, not adjectives.
- Match the CATEGORY VOICE described. It is not a suggestion.
- 2-4 short sentences. Under 320 characters. No links, ever.
- Close with `cta_sentence` in your own words, keeping it low-friction. Put \
something on the table — the number, the offer, the protocol — so that saying \
yes costs them nothing. "Want to look at what changed?" offers nothing; \
"Want the 3-month protocol summary?" does.
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


#: category.voice keys, with the names this dataset actually uses. The old
#: list guessed `taboo_words` and `do_not_say`, neither of which exists in any
#: category pack — so the prompt was getting tone and register and nothing
#: else. `vocab_taboo` is the field the judge checks a message against for
#: promotional or hype register, and it was never reaching the model.
_VOICE_KEYS = (
    "register", "tone", "code_mix",
    "vocab_allowed", "vocab_taboo", "avoid", "taboo_words", "do_not_say",
)

#: Field names for the pack's own worked examples. Every category ships both,
#: and they are the single most direct instruction available: they show the
#: voice rather than describe it. "Quick one — IPL match nights have been 1.5x
#: your weekday avg this season" tells the model more than four lines of
#: adjectives about being warm and practical do.
_EXAMPLE_KEYS = ("tone_examples", "salutation_examples", "message_examples")


def _voice_block(category: dict) -> str:
    """The category's own voice guide, including its worked examples.

    This is the judge's data, not ours. The examples are the pack telling us
    what a message in this category sounds like, written by whoever wrote the
    rest of the pack — which is why they beat anything paraphrased here.
    """
    v = category.get("voice") or {}
    if not isinstance(v, dict):
        return ""
    bits: list[str] = []
    for key in _VOICE_KEYS:
        val = v.get(key)
        if not val:
            continue
        if isinstance(val, (list, tuple)):
            shown = ", ".join(str(x) for x in val[:24])
            if not shown:
                continue
            label = "words to avoid" if "taboo" in key or key in (
                "avoid", "do_not_say") else "words this category uses"
            bits.append(f"  {label}: {shown}")
        else:
            bits.append(f"  {key}: {val}")

    examples: list[str] = []
    for key in _EXAMPLE_KEYS:
        for item in v.get(key) or []:
            text = str(item).strip()
            if text and text not in examples:
                examples.append(text)
    if examples:
        bits.append("  examples of this category's voice (match the shape, "
                    "never copy a fact from them):")
        bits.extend(f"    - {e}" for e in examples[:4])

    return "CATEGORY VOICE\n" + "\n".join(bits) if bits else ""


def _fact_line(label: str, value: Any) -> str:
    text = str(value or "").strip()
    return f"  {label}: {text}" if text else ""


#: How to apply each compulsion lever, worded from the challenge brief's own
#: list (section 10) rather than invented here. The brief is explicit that
#: "a diversified conversation portfolio wins" and that social proof and
#: asking-the-merchant "barely fire" in production — so the mapping is
#: deliberate, not decorative.
_LEVER_GUIDANCE = {
    "specificity": "Anchor on a number, date or source they can check. No adjectives.",
    "loss_aversion": "Name what they are losing or what closes: 'before this window shuts'.",
    "social_proof": "Show that others like them did this: peers in the same category or locality.",
    "effort_externalization": "Do the work first, then ask for a one-word go-ahead.",
    "curiosity": "Open a gap worth closing: 'want to see which ones?'",
    "reciprocity": "Lead with what you noticed on their behalf — a favour, not a pitch.",
    "asking_merchant": "Ask them something only they can answer, and make it easy.",
    "urgency": "Say plainly how little time there is, without manufacturing panic.",
}


def build_prompt(
    signal: Signal,
    category: dict,
    merchant: dict,
    customer: Optional[dict] = None,
    already_sent: Optional[list[str]] = None,
    cta_sentence: str = "",
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
        # The ask this message exists to make, verbatim. The signal carries a
        # `question` for exactly the families where the question IS the CTA,
        # and it used to be dropped on the floor here: `the_point` prefers the
        # headline, so a signal carrying both sent the model a finding and no
        # question at all. It then wrote a message that stated the finding
        # and never asked — the shape the judge scores as no decision made.
        "question_to_ask": signal.question,
        # The exact closing ask, from the same family table the deterministic
        # composer uses. Shipping it means the two paths end the message the
        # same way, and it gives the numbers inside it ("Takes 5 min", a slot
        # menu, "Reply 1 for...") a source the grounding check can see.
        "cta_sentence": cta_sentence,
        # The persuasion strategy chosen upstream, with the brief's own
        # wording for it. This used to stop at the Signal: 24 signals assign
        # a lever, the ranking reads it, and it never reached the model — so
        # every message came out in the same generic register and the
        # engagement plan was computed and thrown away.
        "compulsion_lever": signal.lever,
        "how_to_apply_it": _LEVER_GUIDANCE.get(signal.lever, ""),
        "audience": ("a customer of this business" if signal.customer_facing
                     else "the business owner"),
    }

    biz = g.dig(merchant, "identity", "name", default="") or ""
    area = ", ".join(
        x for x in [g.dig(merchant, "identity", "locality", default=""),
                    g.dig(merchant, "identity", "city", default="")] if x)
    # Both audiences get told which business is writing. For a customer-facing
    # message this is what stops the model inventing one: the merchant's
    # identity used to be absent from the customer prompt entirely, and a
    # refill reminder for Apollo Pharmacy went out signed "Dr. Meera" — a
    # doctor at a different clinic, named nowhere in the context, opening a
    # message to a patient who had never heard of her.
    facts["their_business"] = biz
    facts["their_area"] = area

    # The performance window is context, not a fact to be avoided. Every
    # merchant counter here is a 30-day figure, and a model that says "1,200
    # views in 30d" is being precise rather than inventing — but the number 30
    # appeared in no fact it had been given, so the grounding check read it as
    # a fabrication and threw the message away. Naming the window makes the
    # claim checkable instead.
    window = g.dig(merchant, "performance", "window_days", default=None)
    if isinstance(window, (int, float)) and window > 0:
        facts["their_numbers_window"] = f"last {int(window)} days"

    if customer:
        recipient = g.customer_first_name(customer)
        facts["recipient_first_name"] = recipient
        facts["recipient_state"] = g.dig(customer, "state", default="")
        facts["language_preference"] = g.dig(
            customer, "identity", "language_pref", default="")
        if not recipient:
            # The line above renders as nothing when the name is unknown, and
            # an absent fact is indistinguishable from an oversight — the model
            # fills the gap. Say it out loud instead: this name is not
            # available, attribute the business.
            facts["greeting_rule"] = (
                "recipient_first_name is NOT known — do not invent a name or a "
                "title. Open by naming their_business instead.")
    else:
        # Merchant-facing. The recipient's name has to be supplied explicitly.
        # Left out, the model has no way to greet them and silently writes an
        # impersonal note that reads as a bulk broadcast — which is exactly
        # what the judge's merchant_fit dimension punishes.
        facts["recipient_first_name"] = g.merchant_first_name(merchant)

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
# provider reliability
# --------------------------------------------------------------------------

#: Total attempts per message, i.e. one initial call plus at most
#: LLM_MAX_ATTEMPTS - 1 retries. Small on purpose: a 10-action tick runs its
#: calls in parallel, so every retry is also latency on the judge's clock.
LLM_MAX_ATTEMPTS = 2

#: Cap on any single backoff, whatever Retry-After asks for. A provider that
#: says "come back in 60s" is telling us something we cannot act on inside a
#: 30s endpoint budget, so the wait is clamped and the message falls back.
LLM_MAX_BACKOFF_S = 2.0

#: Statuses worth a second attempt. 429 means the request was well-formed and
#: the provider was momentarily full; 5xx is a transient server fault. 4xx
#: otherwise is our own fault (bad key, bad model) and retrying only burns
#: the tick's budget before falling back to the same answer.
_RETRYABLE = frozenset({429, 500, 502, 503, 504, 529})

#: Counters for /v1/healthz. A silent degrade to the deterministic composer
#: looks identical to a good run from the outside — the bot still answers,
#: still never 500s — which is exactly why a provider problem went unnoticed.
#: These make the difference visible without logging a single message body.
_STATS_LOCK = threading.Lock()
#: Per-thread tick deadline, set by compose_many() for the duration of a job.
_CTX = threading.local()
_STATS = {
    "llm_ok": 0,
    "fallback": 0,
    "cached": 0,
    "rate_limited": 0,
    "retries": 0,
    "provider_error": 0,
    "timeout": 0,
    "no_provider": 0,
    #: Model output refused before it could ship: a number that is in no fact
    #: it was given, or an opening that addresses someone who is not in the
    #: conversation. Counted because it is the difference between "the model
    #: is writing the prose" and "the model writes prose and we throw most of
    #: it away", and those look identical from outside.
    "rejected": 0,
}


def stats() -> dict:
    with _STATS_LOCK:
        return dict(_STATS)


def reset_stats() -> None:
    with _STATS_LOCK:
        for k in _STATS:
            _STATS[k] = 0


def _bump(key: str, n: int = 1) -> None:
    with _STATS_LOCK:
        _STATS[key] = _STATS.get(key, 0) + n


def _note(reason: str, transient: bool = False) -> None:
    """Record why a composition fell back, without logging any content."""
    if transient:
        _bump("rate_limited" if reason.startswith("http 429") else "provider_error")
    print(f"        [llm] fallback: {reason}")


def _retryable(status: int) -> bool:
    return status in _RETRYABLE


def _retry_delay(r, attempt: int) -> float:
    """Honour Retry-After when it is something we can actually wait out."""
    wait = LLM_MAX_BACKOFF_S
    try:
        raw = r.headers.get("Retry-After")
    except Exception:  # noqa: BLE001
        raw = None
    if raw:
        try:
            wait = min(wait, float(str(raw).strip()))
        except (TypeError, ValueError):
            pass  # a date-form Retry-After is not a number of seconds
    else:
        wait = min(wait, 0.4 * (2 ** attempt))
    return max(0.0, wait)


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------


def generate(bundle: PromptBundle, deadline: Optional[float] = None) -> Optional[str]:
    """Call the configured model. Returns None if unavailable or on any failure.

    Returning None rather than raising is load-bearing: the judge treats a 500
    as a hard failure, so a provider outage, a rate limit or a malformed
    response must degrade to the deterministic composer, never to an error.
    """
    if not config.llm_configured():
        return None

    if deadline is None:
        # compose_many() stashes the tick deadline on the worker thread so a
        # retry here can ask whether there is still budget to spend.
        deadline = getattr(_CTX, "deadline", None)

    cached = cache_get(bundle)
    if cached is not None:
        _bump("cached")
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
            # 429 is the one failure worth retrying: it means the request was
            # fine and the provider was momentarily full. We retry on the
            # server's own terms where it gives them, and on a short backoff
            # where it does not — but only ever inside the tick deadline
            # already computed above, because a slow tick is a worse outcome
            # than a fallback body. Everything else degrades immediately.
            for attempt in range(1, LLM_MAX_ATTEMPTS):
                if not _retryable(r.status_code):
                    break
                wait = _retry_delay(r, attempt)
                if deadline is not None and _remaining(deadline) < wait + 1.5:
                    # No budget left to make the retry worthwhile. Stopping
                    # here still leaves time for the deterministic composer.
                    break
                _bump("retries")
                time.sleep(wait)
                r = client.post(url, json=payload,
                                headers=_headers(config.LLM_PROVIDER, key))
            if r.status_code >= 400:
                _note(f"http {r.status_code}", r.status_code in _RETRYABLE)
                return None
            data = r.json()
    except Exception:  # noqa: BLE001 — any failure must degrade, never 500
        _note("exception", True)
        return None

    body = parse_body(_extract_text(config.LLM_PROVIDER, data)) or None
    if body:
        _bump("llm_ok")
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
        # The deadline has to reach generate() so a retry can ask whether
        # there is still budget. compose() takes no deadline argument, and
        # threading one through every layer would change a signature the
        # composer and its tests both depend on — so each pool thread carries
        # it for the duration of its own job.
        _CTX.deadline = deadline
        try:
            results[idx] = _compose_one(*job)
        finally:
            _CTX.deadline = None

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
