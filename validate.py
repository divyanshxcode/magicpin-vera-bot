"""Post-composition validation.

A message that trips one of these is worse than no message at all, so the
composer checks its own output and the tick layer falls back to a safe
deterministic variant rather than shipping something the judge will punish.

Hard failures come from the brief and the API examples:
  - URLs in body        → −3, "Meta would reject"
  - same body twice     → −2 anti-repetition
  - empty body          → −2 malformed
  - hallucinated facts  → −2, the fastest way to get rejected
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

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

    if not cleaned:
        issues.append("body emptied by cleaning")

    # Non-blocking warnings the composer can live with.
    if not re.search(r"\d", cleaned) and not issues:
        issues.append("warning: no number in body (specificity risk)")

    return ValidationResult(not issues, issues, cleaned)
