"""Post-composition validation.

A message that trips one of these is worse than no message at all, so the
composer checks its own output and the tick layer falls back to a safe
deterministic variant rather than shipping something the judge will punish.

Hard failures come from the brief and the API examples:
  - URLs in body        → −3, "Meta would reject"
  - same body twice     → −2 anti-repetition
  - empty body          → −2 malformed
  - hallucinated facts  → −2, the fastest way to get rejected
  - wrong recipient     → the merchant_fit dimension, scored every message

Two of those checks need more than the body to run, so they live here as
standalone functions the composer calls on model output before it is accepted:

  `ungrounded_numbers` — every number in the message must trace to the FACTS
                         block the model was given. A number that appears
                         nowhere is the fabrication the brief penalises.
  `opening_ok`         — the message must open with the person it is for: the
                         recipient's name, or the business when the
                         recipient's name is genuinely unknown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import grounding as g

# Deliberately broad. Any http(s):// or bare domain is a risk, and the judge
# never *requires* a link — "no hard cap, include links only when they add real
# value" is permission, not a requirement.
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[a-z0-9-]+\.(?:com|net|org|in|co\.in|io|ai|app|shop|store|co)\b(?:/\S*)?",
    re.IGNORECASE,
)

MULTI_CTA_RE = re.compile(
    r"\b(reply\s+\S+\s+for\b.*\breply\b|"
    r"reply\s+\w+\s+(?:or|to)\b.*\breply\b)",
    re.IGNORECASE,
)

PREAMBLE_RE = re.compile(
    r"^(?:i hope|hope you(?:'re| are)|good (?:morning|afternoon|evening))\b"
    r"|^just (?:checking|wondering)\s+to\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# number grounding
# --------------------------------------------------------------------------

#: A number as a person writes one. Digit groups, optional decimal, and the
#: currency/percent/multiplier markers that sit on top of it are excluded by
#: the lookarounds, so "₹4,999" and "38%" and "19:30" all reduce to the digits
#: themselves and compare against the same extraction of the facts.
NUMBER_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:[%xX])?(?![\w])")

#: A 24-hour clock time, matched separately so its 12-hour equivalent counts
#: as the same fact. A model writing "7:30pm" for a 19:30 start has not
#: invented anything; a different hour has.
CLOCK_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")

#: Greetings that may stand in front of a name without breaking the opening
#: check. "Dr." and "Mr." are honorifics, not names, so they are stripped
#: before the name is looked for.
_OPENING_FILLER = re.compile(
    r"^(?:hi|hey|hello|namaste|namaskar|good\s+(?:morning|afternoon|evening)|"
    r"dear|mr|mrs|ms|dr|shri|smt)\b[\s.,!:-]*",
    re.IGNORECASE,
)

#: Titles and honorifics whose trailing period is part of the word. Without
#: this, "Dr. Meera, ..." ends its opening at "Dr" — the name is then missing
#: from the opening we inspect, and every message from a doctor or a
#: pharmacist is rejected for a greeting it did address.
_TITLE_RE = re.compile(
    r"\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|approx|no)\.", re.IGNORECASE)

#: Where a sentence may end. A period only counts when the word before it is
#: not an abbreviation, and a "!" or "?" or a spaced dash always ends the
#: sentence.
_SENTENCE_END_RE = re.compile(r"[.!?]|\s—\s")


def _opening_text(body: str) -> str:
    """The first sentence, or the first few words of the first long one."""
    text = (body or "").strip()
    if not text:
        return ""
    cut = -1
    for match in _SENTENCE_END_RE.finditer(text):
        if match.group(0) != ".":
            cut = match.end()
            break
        # A period closes the sentence unless it belongs to a title: what
        # matters is the word it terminates, so look at what precedes it.
        word = re.split(r"[\s,;:!?]", text[:match.start()])[-1] if text[:match.start()] else ""
        if not _TITLE_RE.fullmatch(f"{word}."):
            cut = match.end()
            break
    if 0 < cut <= _OPENING_CHARS:
        return text[:cut]
    return text[:_OPENING_CHARS]

#: How much of the opening is read before deciding the addressee is missing.
#: Long enough to survive a hook ("Quick heads-up Suresh — ...") and short
#: enough that a name in the third sentence is not an opening.
_OPENING_CHARS = 110


def _normalise_number(token: str) -> Optional[float]:
    """`1,150` -> 1150.0, `04` -> 4.0, `2.5` -> 2.5. None if unparseable."""
    try:
        value = float(token.replace(",", ""))
    except (TypeError, ValueError):
        return None
    return value


def _numbers_in(text: str) -> list[float]:
    out: list[float] = []
    for match in NUMBER_RE.finditer(text or ""):
        value = _normalise_number(match.group(1))
        if value is not None:
            out.append(value)
    return out


def _allowed_numbers(facts: Optional[dict], extra: Iterable[str] = ()) -> set[float]:
    """Every number the facts legitimately license, in any rendering.

    Facts are written for humans, so the same fact arrives several ways and
    each is a legitimate thing for the model to say:

      - "196 days"                  → 196
      - "2026-04-28"                → 2026, 4, 28 (a date, read componentwise)
      - "19:30"                     → 19, 30, and 7 (7:30pm is the same fact)
      - "0.048"                     → 0.048 and 4.8 (a rate read as a percent)
      - ["Wed 5 Nov, 6pm"]          → 5, 6, and len(list) == 1

    The point is to make the check catch a number that is nowhere in the
    prompt, without making it so literal that it rejects the facts it was
    given back in a different notation.
    """
    allowed: set[float] = set()

    def take(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            v = float(value)
            allowed.add(v)
            # A rate stored as a fraction is the same number written as a
            # percentage, which is how a merchant reads it out loud.
            if v and abs(v) < 1:
                allowed.add(round(v * 100, 6))
            return
        if isinstance(value, (list, tuple)):
            allowed.add(float(len(value)))
            for item in value:
                take(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                take(item)
            return
        for number in _numbers_in(str(value)):
            allowed.add(number)
        for hour, minute in CLOCK_RE.findall(str(value)):
            allowed.add(float(hour))
            allowed.add(float(minute))
            twelve = int(hour) % 12
            if twelve:
                allowed.add(float(twelve))

    for value in (facts or {}).values():
        take(value)
    for text in extra:
        take(text)
    return allowed


def ungrounded_numbers(body: str, facts: Optional[dict],
                       extra: Iterable[str] = ()) -> list[str]:
    """Numbers in `body` that appear nowhere in the facts, rendered for a log.

    A model that rephrases a fact says the same number. A model that invents
    one says a number that was never offered to it — and the judge scores that
    as fabrication, not as creativity. The caller's job on a non-empty result
    is to throw the body away, not to edit it: a message with one number too
    many is not a message worth patching.
    """
    if not body:
        return []
    allowed = _allowed_numbers(facts, extra)
    if not allowed:
        return []
    seen: list[str] = []
    for value in _numbers_in(body):
        if value in allowed or int(value) in allowed:
            continue
        # A clock hour is the same fact in either clock: "19:30" licenses
        # "7:30pm", so a bare 7 with a 30 beside it is not a new number.
        if value in allowed:
            continue
        text = f"{value:g}"
        if text not in seen:
            seen.append(text)
    return seen


def _bare(text: str) -> str:
    """Lowercased, punctuation stripped, so names compare cleanly."""
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())


def _has_name(head: str, name: str) -> bool:
    bare = _bare(name)
    if not bare:
        return False
    if bare in head:
        return True
    # "Mr. Sharma" is satisfied by a message that says only "Sharma", and a
    # business name by one that uses its distinctive first word. Requiring the
    # whole string rejects messages that are merely less formal, not wrong.
    tokens = [t for t in bare.split() if len(t) >= 3]
    return any(t in head for t in tokens)


def opening_ok(body: str, recipient: str = "", business: str = "",
               customer_facing: bool = False) -> Optional[str]:
    """None when the opening is right, else the reason it is not.

    The judge reads merchant fit as "uses their name correctly" and scores it
    on every message. Two ways to get that wrong are fixed here:

      - addressing someone who is not in the conversation. A customer-facing
        message that opens "Dr. Meera" for a pharmacy has greeted a stranger
        who appears nowhere in the context the model was given.
      - addressing nobody at all. A message that opens on a fact and never
        names its reader reads as a broadcast, and the judge says so in as many
        words.

    So the name goes first. Where the recipient's name is genuinely unknown —
    a customer id that says "grandfather" rather than a person, a merchant
    record with no owner — the business itself is the addressee, because a
    message on the merchant's behalf has to say whose behalf it is on.
    """
    head = _bare(_opening_text(body))
    if not head:
        return "empty body"

    if recipient:
        stripped = _OPENING_FILLER.sub("", head).strip()
        if not _has_name(stripped, recipient) and not _has_name(stripped, business):
            return f"does not open with the recipient's name ({recipient})"
        return None

    if customer_facing and business:
        if not _has_name(head, business):
            return f"recipient unknown and the message never says {business!r}"
        return None

    # Merchant-facing with no owner on record: a message that leads with a
    # finding is the correct shape here, so there is no name to insist on.
    return None



@dataclass
class ValidationResult:
    ok: bool
    issues: list[str]
    cleaned_body: str


def validate(
    body: str,
    category: dict,
    merchant: dict,
    sent_bodies: Optional[set[str]] = None,
    recipient: str = "",
    business: str = "",
    customer_facing: bool = False,
) -> ValidationResult:
    """Check a composed body. Returns issues plus a cleaned body."""
    issues: list[str] = []
    cleaned = (body or "").strip()

    if not cleaned:
        return ValidationResult(False, ["empty body"], "")

    if len(cleaned) > 1200:
        issues.append(f"body too long ({len(cleaned)} chars)")
        cleaned = cleaned[:1197].rstrip() + "..."

    if URL_RE.search(cleaned):
        issues.append("URL in body (hard penalty, Meta would reject)")
        cleaned = URL_RE.sub("", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    taboos = g.voice_taboos(category)
    for taboo in taboos:
        # Only flag taboo phrases that actually appear as a phrase; several
        # entries in the dataset are long enough to never match literally.
        t = taboo.strip().lower()
        if len(t) >= 8 and re.search(rf"\b{re.escape(t)}\b", cleaned, re.IGNORECASE):
            issues.append(f"taboo phrase from category voice: '{taboo}'")
            cleaned = re.sub(rf"\b{re.escape(t)}\b", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    if PREAMBLE_RE.match(cleaned):
        issues.append("long preamble opener")

    if MULTI_CTA_RE.search(cleaned):
        issues.append("multiple CTAs in one message")

    if sent_bodies and cleaned in sent_bodies:
        issues.append("verbatim repeat of a body already sent in this conversation")

    # An opening with no name reads as a bulk broadcast. The Gemini judge
    # scored merchant_fit 5/10 across the board for exactly this, while
    # messages that led with a fact the merchant owns scored 7-9.
    #
    # The addressee is passed in rather than inferred from the merchant,
    # because on a customer-facing message the owner is the wrong person to
    # be looking for: "Hi Kavya — this is Studio11" is addressed to a customer,
    # and checking it against the owner flagged every such message as a
    # failure and wrote the complaint into the rationale. `composer` computes
    # the real recipient; this only reports on it, because by the time a body
    # reaches validate() the hard guard has already run and only the
    # edge cases (a deterministic body, a wording we did not predict) land
    # here.
    problem = opening_ok(cleaned, recipient, business, customer_facing)
    if problem:
        issues.append(f"body {problem}")

    if not cleaned:
        issues.append("body emptied by cleaning")

    # Non-blocking warnings the composer can live with.
    if not re.search(r"\d", cleaned) and not issues:
        issues.append("warning: no number in body (specificity risk)")

    return ValidationResult(not issues, issues, cleaned)
