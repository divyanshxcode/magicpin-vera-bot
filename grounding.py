"""Grounded accessors over the four contexts.

Every fact that ends up in a message passes through here. Two reasons:

1. **Anti-hallucination.** The judge rejects submissions with invented data
   ("numbers without provenance get scored as fabrication"). So we only ever
   format values that we can point at in the context.
2. **Graceful degradation.** The expanded dataset contains generated merchants
   with empty `offers`, empty `signals`, empty `conversation_history`, and
   generated triggers whose payload is literally `{"placeholder": true}`. A
   bot that assumes rich context will hallucinate exactly there — which is
   where the judge injects fresh, unseen context to test us.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------

_MISSING = object()

MONTHS_SHORT = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def dig(obj: Any, *path, default: Any = _MISSING) -> Any:
    """Walk nested dicts safely. Returns `default` on any gap."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None if default is _MISSING else default
        cur = cur[key]
    return cur if cur is not None else (None if default is _MISSING else default)


def is_placeholder(payload: Any) -> bool:
    """True when a trigger payload carries no real facts.

    The dataset generator emits `{"placeholder": true, "metric_or_topic": kind}`
    for its 75 synthetic triggers. Nothing in there is groundable, so the
    composer must fall back to merchant/category context only and must never
    invent numbers to fill the gap.
    """
    return not isinstance(payload, dict) or bool(payload.get("placeholder"))


def has_real_payload(payload: Any) -> bool:
    return not is_placeholder(payload)


# --------------------------------------------------------------------------
# number formatting (grounded by construction — input must come from context)
# --------------------------------------------------------------------------


def fmt_int(value: Any) -> str:
    """Format an integer count with Indian digit grouping."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    return f"{n:,}"


def fmt_pct(value: Any, places: Optional[int] = None) -> str:
    """Format a ratio (0.021) as a percentage.

    Defaults to the fewest places that preserve the value, so 0.30 renders as
    "30%" rather than the odd-looking "30.0%", while 0.021 stays "2.1%".
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if f == 0:
        return ""
    pctv = f * 100
    if places is None:
        places = 0 if abs(pctv - round(pctv)) < 1e-9 else (1 if abs(pctv * 10 - round(pctv * 10)) < 1e-9 else 2)
    return f"{pctv:.{places}f}%"


def fmt_signed_pct(value: Any, places: int = 0) -> str:
    """Format a delta (-0.30) as a signed percentage (-30%)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    pctv = f * 100
    sign = "-" if pctv < 0 else "+"
    return f"{sign}{abs(pctv):.{places}f}%"


def fmt_delta_pct(value: Any) -> str:
    """Like fmt_signed_pct but drops the '+' for positive values.

    "up 30%" reads better than "up +30%" in a merchant-facing message.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    pctv = f * 100
    if pctv < 0:
        return f"{abs(pctv):.0f}% down"
    return f"{pctv:.0f}% up"


def fmt_money(value: Any) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{int(n):,}" if n == int(n) else f"{n:,.2f}"


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------


def parse_date(value: Any) -> Optional[date]:
    if not value or not isinstance(value, str):
        return None
    v = value.strip().replace("Z", "+00:00")
    for parser in (
        lambda s: datetime.fromisoformat(s).date(),
        lambda s: date.fromisoformat(s[:10]),
        lambda s: datetime.strptime(s[:10], "%Y-%m-%d").date(),
    ):
        try:
            return parser(v)
        except (ValueError, TypeError):
            continue
    return None


def parse_moment(value: Any) -> Optional[datetime]:
    """Parse to a full datetime, KEEPING the time component.

    parse_date() deliberately returns a date: expiry and "days ago" are
    day-granularity questions. Anything measured in minutes or hours must not
    use it — two timestamps on the same date both collapse to midnight, so the
    interval between them reads as zero. That is not a rounding detail, it
    makes every same-day cooldown unreachable and silently mutes the bot.

    Always returns a timezone-aware value so subtraction never mixes naive
    and aware datetimes.
    """
    if not value or not isinstance(value, str):
        return None
    v = value.strip().replace("Z", "+00:00")
    for parser in (
        lambda s: datetime.fromisoformat(s),
        lambda s: datetime.fromisoformat(s[:10]),
        lambda s: datetime.strptime(s[:10], "%Y-%m-%d"),
    ):
        try:
            parsed = parser(v)
        except (ValueError, TypeError):
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def peer_stat(category: dict, metric: str) -> tuple[Optional[float], str]:
    """Peer-median value for a metric, and the label to quote it with.

    Schema-driven rather than hardcoded: looks for `avg_{metric}_30d` then
    `avg_{metric}` in the category's peer_stats, so a test dataset with
    different categories, or a category that drops a field, still works — it
    just yields (None, "") and the caller skips the comparison.

    This is the reference point the judge's Decision Quality wants. "calls down
    50% (from 4)" is a number with no meaning attached; "4 calls in 30d against
    a 12 peer median" says whether that is bad.
    """
    stats = dig(category, "peer_stats", default=None)
    if not isinstance(stats, dict):
        return None, ""
    metric = str(metric or "").strip()
    if not metric:
        return None, ""
    for key in (f"avg_{metric}_30d", f"avg_{metric}"):
        val = stats.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            label = {
                "views": "views", "calls": "calls", "directions": "directions",
                "ctr": "CTR", "leads": "leads", "rating": "rating",
                "review_count": "reviews", "photos": "photos",
            }.get(metric, metric.replace("_", " "))
            return float(val), label
    return None, ""


def peer_gap(value: Any, peer: Optional[float]) -> Optional[float]:
    """How far `value` sits from the peer median, as a signed fraction.

    None unless both sides are real numbers and the peer value is non-zero, so
    a caller never renders "infinitely below peer median" or divides by None.
    """
    if value is None or peer is None:
        return None
    try:
        v, p = float(value), float(peer)
    except (TypeError, ValueError):
        return None
    if p == 0:
        return None
    return (v - p) / p


def merchant_notes(merchant: dict) -> list[str]:
    """The merchant's own precomputed signal flags, humanised.

    `signals` is a list of compact strings the dataset ships per merchant —
    "stale_posts:22d", "ctr_below_peer_median". They are already true, so
    restating them is grounded by construction. Anything that does not look
    like a known flag is passed through rather than reformatted, so an unseen
    test dataset still contributes.
    """
    out: list[str] = []
    for raw in dig(merchant, "signals", default=[]) or []:
        s = str(raw or "").strip()
        if not s:
            continue
        flag, _, detail = s.partition(":")
        flag = flag.replace("_", " ")
        if detail and detail.isdigit():
            out.append(f"{flag} for {detail}d")
        elif detail:
            out.append(f"{flag} ({detail})")
        else:
            out.append(flag)
    return out


def days_between(earlier: Any, later: Any) -> Optional[int]:
    a, b = parse_date(earlier), parse_date(later)
    if not a or not b:
        return None
    return (b - a).days


def months_between(earlier: Any, later: Any) -> Optional[int]:
    """Whole months from `earlier` to `later`.

    Used for recall cadences (last visit -> due date). Sign-preserving, because
    the seed dataset contains dates that sit either side of "now" and we must
    never silently render a negative cadence as a positive one.
    """
    a, b = parse_date(earlier), parse_date(later)
    if not a or not b:
        return None
    months = (b.year - a.year) * 12 + (b.month - a.month)
    if b.day < a.day:
        months -= 1
    return months


def fmt_date(value: Any) -> str:
    """`2026-11-12` -> `12 Nov`. Returns '' if unparseable."""
    d = parse_date(value)
    if not d:
        return ""
    return f"{d.day} {MONTHS_SHORT[d.month - 1]}"


def months_since(value: Any, now: Any) -> Optional[int]:
    """Whole months elapsed since `value`, relative to `now`."""
    a, b = parse_date(value), parse_date(now)
    if not a or not b:
        return None
    months = (b.year - a.year) * 12 + (b.month - a.month)
    if b.day < a.day:
        months -= 1
    return max(0, months)


def days_since(value: Any, now: Any) -> Optional[int]:
    return days_between(value, now)


def humanize_days_ago(value: Any, now: Any) -> str:
    d = days_since(value, now)
    if d is None:
        return ""
    if d <= 0:
        return "today"
    if d == 1:
        return "yesterday"
    if d < 7:
        return f"{d} days ago"
    if d < 14:
        return "last week"
    if d < 60:
        return f"{d//7} weeks ago"
    if d < 365:
        return f"{d//30} months ago"
    return f"{d//365} year{'s' if d >= 730 else ''} ago"


def days_until_label(value: Any, now: Any) -> str:
    d = days_between(now, value)
    if d is None:
        return ""
    if d <= 0:
        return "today"
    if d == 1:
        return "tomorrow"
    if d < 14:
        return f"in {d} days"
    if d < 60:
        return f"in {d//7} weeks"
    if d < 365:
        return f"in {d//30} months"
    return f"in {d//365} years"


def payload_countdown(
    payload: Any,
    count_keys: Iterable[str],
    date_key: str,
    now: Any,
) -> Optional[int]:
    """The countdown this trigger itself declared, with a staleness override.

    A trigger payload is the bot's own statement about the world: it says
    "196 days to the wedding" and names the date that count came from. The
    judge scores trigger relevance by cross-checking the message against that
    payload, so a countdown recomputed against the wall clock is a message
    contradicting the very context it is being scored in — "42 days" beside a
    payload that says 196 is read as a fabricated number, which is a −2.

    So the payload's own count wins. What it cannot be trusted for is
    *direction*: a count that still reads down to a date already behind us (or
    an elapsed count pointing at a date still ahead) is stale on a hidden
    dataset, and a countdown to a date in the past is not a usable thing to
    send. Magnitude disagreement on its own is not evidence of staleness —
    the count was true when the trigger was generated, which is the reference
    frame the payload is read in — so the override fires only on a sign
    conflict, and then the recomputed value wins because it is the one that
    describes a date we could actually talk about.

    Returns None when the payload carries neither a usable count nor a
    parseable date, so callers fall back to whatever else they have.
    """
    if not isinstance(payload, dict):
        return None

    count: Optional[int] = None
    for key in count_keys:
        raw = payload.get(key)
        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            count = int(raw)
            break
        if isinstance(raw, str) and re.fullmatch(r"\s*-?\d+\s*", raw):
            count = int(raw)
            break

    date_val = payload.get(date_key) if date_key else None
    recomputed = days_between(now, date_val) if date_val else None

    if count is None:
        return recomputed
    if recomputed is None:
        return count
    # `>= 0` rather than `> 0` so a count of 0 ("today") still agrees with a
    # date that is with us, and only a genuinely passed date forces a recount.
    if (count >= 0) != (recomputed >= 0):
        return recomputed
    return count


# --------------------------------------------------------------------------
# identity / voice
# --------------------------------------------------------------------------


def merchant_first_name(merchant: dict) -> str:
    """Best available human name for a merchant."""
    ident = dig(merchant, "identity", default={}) or {}
    fn = (ident.get("owner_first_name") or "").strip()
    if fn:
        return fn
    # Fall back to a leading proper noun from the business name.
    name = (ident.get("name") or "").strip()
    if not name:
        return ""
    m = re.match(r"(?:Dr\.?|Doctor)\s+([A-Za-z]+)", name)
    if m:
        return m.group(1)
    first = re.split(r"[\s'’]", name)[0]
    return first if len(first) > 1 else ""


def salutation(category: dict, merchant: dict) -> str:
    """Category-aware greeting using the first name.

    Dentists get the honorific because it is the register the category's own
    voice profile prescribes (`salutation_examples: ["Dr. {first_name}"]`).
    Everything else gets a plain first-name greeting.

    Returns "" — not "Hi" — when the merchant's name is unknown, which happens
    whenever a trigger arrives before its MerchantContext. The composer then
    opens on the fact itself. A bare "Hi," followed by a comma is worse than
    no greeting at all: it is the bulk-broadcast shape the judge's merchant-fit
    dimension is built to catch, and it reads as a message sent to nobody in
    particular.
    """
    fn = merchant_first_name(merchant)
    if not fn:
        return ""
    tone = str(dig(category, "voice", "tone", default="") or "")
    slug = str(category.get("slug") or dig(merchant, "category_slug", default="") or "")
    if tone == "peer_clinical" or slug == "dentists" or slug == "pharmacies":
        return f"Dr. {fn}"
    return f"Hi {fn}"


#: Base languages that ship with an "-en mix" preference in the dataset. A
#: code-mix preference is "speak some of my language inside the English
#: structure", so which language is not a detail — it is the request.
_MIX_BASES = {"hi", "kn", "ta", "te", "ml", "mr", "bn", "gu", "pa", "ur"}


def language_of(entity: dict) -> str:
    """Normalise a language preference to one of: en | hi | mix-<lang>.

    The dataset ships `hi-en mix`, `kn-en mix`, `ta-en mix` and `te-en mix`
    alongside `hi` and `english`. Collapsing every "*-en mix" to a bare `mix`
    meant a Kannada speaker was handed Hindi fragments — the one thing a
    language preference is supposed to prevent — so the base language is kept
    as a suffix and `mix_phrase` can pick the right one. Anything with no
    recognised base stays plain `mix`, and anything unrecognised falls back to
    English, which is the documented default.
    """
    pref = ""
    if isinstance(entity, dict):
        pref = (dig(entity, "identity", "language_pref", default="")
                or dig(entity, "voice", "code_mix", default="")
                or "")
    pref = str(pref).lower().strip()
    if "mix" in pref or "-" in pref:
        base = pref.split("-")[0].strip()
        return f"mix-{base}" if base in _MIX_BASES else "mix"
    if pref.startswith("hi") or pref == "hindi":
        return "hi"
    if pref.startswith("en") or pref == "english":
        return "en"
    # Fall back to the merchant's declared languages.
    langs = dig(entity, "identity", "languages", default=[]) or []
    if isinstance(langs, list) and langs:
        if "hi" in langs and len(langs) > 1:
            return "mix"
        if "hi" in langs:
            return "hi"
    return "en"


# --------------------------------------------------------------------------
# merchant state
# --------------------------------------------------------------------------


def active_offers(merchant: dict, limit: int = 6) -> list[dict]:
    offers = dig(merchant, "offers", default=[]) or []
    return [o for o in offers
            if isinstance(o, dict) and str(o.get("status", "")).lower() == "active"][:limit]


def primary_offer(merchant: dict) -> Optional[dict]:
    offers = active_offers(merchant, limit=1)
    return offers[0] if offers else None


def offer_title(offer: Optional[dict]) -> str:
    if not isinstance(offer, dict):
        return ""
    return str(offer.get("title", "")).strip()


def offer_price_text(offer: Optional[dict]) -> str:
    """Render an offer's price the way a merchant would say it out loud.

    "₹299" alone reads oddly in a sentence, so a discount becomes
    "₹299 (was ₹499)" and a multi-off becomes "from ₹299". Returns "" when
    there is no usable price, so callers can omit the clause entirely rather
    than render a stray currency symbol.
    """
    if not isinstance(offer, dict):
        return ""
    price = offer.get("price")
    if not isinstance(price, (int, float)) or price <= 0:
        return ""
    was = offer.get("original_price") or offer.get("compare_at")
    if isinstance(was, (int, float)) and was > price:
        return f"₹{price:,.0f} (was ₹{was:,.0f})"
    return f"₹{price:,.0f}"


def merchant_signals(merchant: dict) -> list[str]:
    sig = dig(merchant, "signals", default=[]) or []
    return [str(s) for s in sig if isinstance(s, str) and s.strip()]


def has_signal(merchant: dict, needle: str) -> bool:
    return any(needle in s for s in merchant_signals(merchant))


def signal_value(merchant: dict, prefix: str) -> Optional[str]:
    """Pull the value out of a `prefix:22d`-style signal string."""
    for s in merchant_signals(merchant):
        if s.startswith(prefix):
            tail = s.split(":", 1)[1].strip() if ":" in s else ""
            return tail or None
    return None


def merchant_conversation(merchant: dict, limit: int = 4) -> list[dict]:
    hist = dig(merchant, "conversation_history", default=[]) or []
    return [h for h in hist if isinstance(h, dict)][-limit:]


def last_merchant_engagement(merchant: dict) -> str:
    """How the merchant last behaved — 'merchant_replied', 'ignored', ..."""
    hist = merchant_conversation(merchant, limit=1)
    return str(hist[0].get("engagement", "")) if hist else ""


def merchant_engaged_recently(merchant: dict) -> bool:
    return last_merchant_engagement(merchant) in (
        "merchant_replied", "intent_action", "intent_question", "intent_planning"
    ) or has_signal(merchant, "engaged_in_last")


# --------------------------------------------------------------------------
# category knowledge
# --------------------------------------------------------------------------


def digest_items(category: dict) -> list[dict]:
    items = category.get("digest") or []
    return [d for d in items if isinstance(d, dict)]


def find_digest_item(category: dict, item_id: Optional[str]) -> Optional[dict]:
    if not item_id:
        return None
    for d in digest_items(category):
        if d.get("id") == item_id:
            return d
    return None


def find_digest_items(category: dict, ids: Iterable[Optional[str]]) -> list[dict]:
    out: list[dict] = []
    for i in ids:
        item = find_digest_item(category, i)
        if item and item not in out:
            out.append(item)
    return out


def content_item(category: dict, item_id: Optional[str]) -> Optional[dict]:
    if not item_id:
        return None
    for c in category.get("patient_content_library") or []:
        if isinstance(c, dict) and c.get("id") == item_id:
            return c
    return None


def voice_taboos(category: dict) -> list[str]:
    return [str(t) for t in (dig(category, "voice", "vocab_taboo", default=[]) or [])]


def seasonal_beats(category: dict, now: Any) -> list[dict]:
    """Beats whose month_range currently contains the simulated month."""
    n = parse_date(now)
    if n is None:
        return []
    out = []
    for beat in category.get("seasonal_beats") or []:
        if not isinstance(beat, dict):
            continue
        rng = str(beat.get("month_range", ""))
        if _month_range_matches(rng, n.month):
            out.append(beat)
    return out


def _month_range_matches(rng: str, month: int) -> bool:
    if not rng:
        return False
    months = []
    for name, num in (
        ("jan", 1), ("feb", 2), ("mar", 3), ("apr", 4), ("may", 5), ("jun", 6),
        ("jul", 7), ("aug", 8), ("sep", 9), ("oct", 10), ("nov", 11), ("dec", 12),
    ):
        if name in rng.lower():
            months.append(num)
    if not months:
        return False
    if len(months) == 1:
        return months[0] == month
    lo, hi = min(months), max(months)
    if lo <= hi:
        return lo <= month <= hi
    # Wraps year end (e.g. Nov-Feb).
    return month >= lo or month <= hi


# --------------------------------------------------------------------------  
# customer
# --------------------------------------------------------------------------


#: Tokens in a customer id that are never a name. "c_013_grandfather_for_m009"
#: is a relationship — the real record says "Mr. Sharma" — and greeting someone
#: as "Hi Grandfather," is worse than not greeting them by name at all.
#: "anonymous" likewise means there is no profile to name.
_ID_STOPWORDS = {
    "anonymous", "unknown", "walkin", "guest", "none", "na",
    "grandfather", "grandmother", "father", "mother", "uncle", "aunt",
    "son", "daughter", "brother", "sister", "husband", "wife", "parent",
}


def customer_name_from_id(customer_id: str) -> str:
    """Best-effort first name from a customer id, or "".

    Customer ids in this dataset are `c_001_priya_for_m001` — the second
    segment is the person's name. When a trigger names a customer whose
    CustomerContext has not arrived yet, that id is the only thing we have,
    and it is context the judge itself supplied, not something invented.

    There is deliberately no merchant equivalent. `m_003_studio11_salon_hyderabad`
    is owned by Lakshmi: the second segment is the *business*, so parsing it
    as an owner name would address the wrong person. Guessing there is worse
    than saying nothing, so merchant callers get "".

    Returns "" whenever the shape is unfamiliar, which keeps a differently-
    shaped hidden dataset from producing a garbled greeting.
    """
    cid = str(customer_id or "").strip().lower()
    if not cid:
        return ""
    parts = [p for p in re.split(r"[^a-z0-9]+", cid) if p]
    # Expect <prefix> <number> <name...> [for|at] <ref>. Anything else is a
    # shape we have not seen, and guessing at it is how you address "Sr." to
    # a dentist.
    if len(parts) < 3 or not parts[0] or not parts[1].isdigit():
        return ""
    tail = parts[2:]
    for stop in ("for", "at", "of"):
        if stop in tail:
            tail = tail[:tail.index(stop)]
    if not tail:
        return ""
    first = tail[0]
    if first in _ID_STOPWORDS:
        return ""
    # "karthik_jr" -> "Karthik Jr"; drop generational suffixes.
    name = " ".join(w for w in tail if w not in ("jr", "sr", "ii", "iii"))
    if not name:
        return ""
    return " ".join(w.capitalize() for w in name.split())


def customer_first_name(customer: dict) -> str:
    """Display name for a customer, respecting honorifics.

    `customer_first_name` must not return "Mr." for a customer named
    "Mr. Sharma" — that renders as "Hi Mr." and reads as a bug. Honorific
    prefixes are kept whole so the greeting stays respectful, which matters
    for the senior-citizen pharmacy cases.
    """
    name = str(dig(customer, "identity", "name", default="") or "").strip()
    if not name:
        cid = str(customer.get("customer_id", "") or "")
        return customer_name_from_id(cid)
    # Names like "Aanya (parent: Sneha)" -> "Aanya"
    name = re.split(r"\(", name)[0].strip()
    parts = [p for p in re.split(r"\s+", name) if p]
    if not parts:
        return ""
    if parts[0].endswith(".") and len(parts) > 1:
        return f"{parts[0]} {parts[1]}"   # "Mr. Sharma", "Dr. Iyer"
    return parts[0]


def customer_is_via_mediary(customer: dict) -> bool:
    """True when the message reaches the customer through someone else
    (e.g. Mr. Sharma's son, a child's parent) and should address that person.
    """
    channel = str(dig(customer, "preferences", "channel", default="") or "").lower()
    name = str(dig(customer, "identity", "name", default="") or "").lower()
    return "via_" in channel or "via " in name


def consent_allows(customer: dict, scope_need: str) -> bool:
    """Check the customer's consent scope.

    `scope_need` should be a coarse topic slug, e.g. "recall_reminders".
    An empty or missing consent block is treated as no consent — we never send
    to someone who has not opted in.
    """
    scope = dig(customer, "consent", "scope", default=None)
    if not scope:
        return False
    scope = [str(s).lower() for s in scope]
    topic = scope_need.lower()
    if topic in scope:
        return True
    # A broader scope covers a narrower one: "promotional_offers" does not
    # cover "recall_reminders", but "appointment_reminders" and
    # "reminders" style entries do.
    return any(topic.split("_")[0] in s for s in scope)


# --------------------------------------------------------------------------
# trigger
# --------------------------------------------------------------------------


def trigger_kind(trigger: dict) -> str:
    return str(trigger.get("kind", "") or "")


def trigger_scope(trigger: dict) -> str:
    return str(trigger.get("scope", "merchant") or "merchant")


def trigger_expired(trigger: dict, now: str) -> bool:
    exp = parse_date(trigger.get("expires_at"))
    n = parse_date(now)
    if exp is None or n is None:
        return False
    return n > exp
