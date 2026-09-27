"""Turn a chosen Signal into the message body.

The judge's per-dimension guidance drives every decision here:

  Specificity          — the fact leads, with its citation attached
  Category fit         — voice comes from `CategoryContext.voice`, never generic
  Merchant fit         — name, language pref, and their own live offers
  Trigger relevance    — the message says *why now*, not "improve your profile"
  Engagement compulsion — one low-friction ask, externalising the effort

Anti-patterns the brief calls out, each handled explicitly below: generic
"10% off" copy, multiple CTAs, buried CTAs, promotional tone in clinical
categories, hallucinated data, long preambles, re-introducing ourselves, and
URLs in the body (a hard Meta rejection, −3 per the API examples).

Determinism: this module is pure. Same four contexts in, same message out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import config
import grounding as g
from signals import (
    CTA_BINARY,
    CTA_CONFIRM,
    CTA_NONE,
    CTA_OPEN,
    CTA_SLOTS,
    Signal,
)

# --------------------------------------------------------------------------
# voice profiles, keyed by category slug
# --------------------------------------------------------------------------
# These mirror each CategoryContext.voice.tone. The point is that a dentist and
# a pharmacy sound like different people: the judge explicitly penalises
# "promotional tone for categories that need clinical/peer voice".

@dataclass(frozen=True)
class Voice:
    register: str          # short description, echoed in the rationale
    icon: str = ""         # category-appropriate emoji, used sparingly
    clinical: bool = False # suppress hype and emoji entirely


VOICES = {
    "dentists": Voice(register="clinical peer", clinical=True),
    "pharmacies": Voice(register="trustworthy neighbourhood pharmacist", clinical=True),
    "salons": Voice(register="approachable expert", icon="✨"),
    "restaurants": Voice(register="fellow operator"),
    "gyms": Voice(register="coach"),
}

DEFAULT_VOICE = Voice(register="practical peer")


def voice_for(category: dict) -> Voice:
    slug = str(category.get("slug", "")).lower()
    if slug in VOICES:
        return VOICES[slug]
    return DEFAULT_VOICE


# --------------------------------------------------------------------------
# language rendering
# --------------------------------------------------------------------------
# `language_pref` values in the dataset: "hi-en mix", "hi", "english", "te-en
# mix", "kn-en mix", "ta-en mix". For a mix preference we keep the structure
# English (technical terms stay readable) but thread a little of the reader's
# own language through, the way production Vera actually writes and the way
# the 50/50 Case Study 2 anchor does it: "Apke liye 2 slots ready hain: Wed 5
# Nov, 6pm ya Thu…"
#
# A bad code-mix reads worse than none, so the phrases are few, fixed, and
# placed where they read naturally — never appended as a dangling suffix. And
# they are per-language: `kn-en mix` is a Kannada reader, and handing them the
# Hindi line below is the single loudest way to ignore who they are.

MIX_SLOTS = "Aapke liye {n} slots ready hain"
MIX_SLOT_ONE = "Ek slot ready hai"
MIX_OFFER = "{offer} chal raha hai"
MIX_REFILL = "Aapki {n} dawaiyaan khatam ho rahi hain"
MIX_REMINDER = "Yeh ek reminder hai"

#: Per-language overrides for the four that ship in the dataset. English
#: carries the sentence structure, so these are short fragments that read as
#: borrowed rather than as a translation — the register a real code-mixed
#: message has. Only `slots` and `reminder` are translated: the offer title
#: and drug names stay in English, which is exactly how these are written in
#: practice, and translating a molecule name would be a fabrication.
MIX_PHRASES: dict[str, dict[str, str]] = {
    "kn": {"slots": "Nimmu {n} slots ready", "slot_one": "Oku slot ready",
           "reminder": "Ee oru reminder", "running": "chaludukte"},
    "ta": {"slots": "Ungaluku {n} slots ready", "slot_one": "Oru slot ready",
           "reminder": "Idhu oru reminder", "running": "nadakkum"},
    "te": {"slots": "Mekapaki {n} slots ready", "slot_one": "Oka slot ready",
           "reminder": "Edi oru reminder", "running": "chalundi"},
}


def apply_language(body: str, lang: str, family: str, category: dict) -> str:
    """Light-touch code-mix. Never translates technical or legal vocabulary.

    The per-language work happens at the point of composition, in
    `mix_phrase` calls that know which fragment they are writing — mixing a
    finished sentence here would mean pattern-matching it afterwards, and a
    regex that guesses where a language starts is worse than not mixing at
    all. So this stays a seam: it decides whether the reader wants a mix at
    all, and leaves the wording to the composer.
    """
    if not body or not str(lang or "").startswith("mix"):
        return body
    return body


def mix_phrase(kind: str, lang: str = "mix", **kw) -> str:
    """A code-mixed fragment in the reader's own base language.

    Hindi is the default because it is the most common `*-en mix` in the
    dataset and the one the case-study anchor was written in. A base language
    with no translation falls back to the English sentence rather than to
    another language's words — silence reads as neutral, Hindi-in-a-Kannada
    message reads as a wrong recipient.
    """
    base = str(lang or "").split("-")[-1] if str(lang or "").startswith("mix") else "hi"
    # A base language we have no phrase for gets the English lead-in, never
    # another language's words. Falling back to Hindi here would hand a Malayalam
    # reader Hindi and call it honouring their preference — the failure this
    # whole change exists to remove. A bare "mix" (no base recorded) still
    # resolves to Hindi, which is the case-study register and the common case.
    if base == "hi":
        table = {}          # the module-level Hindi defaults, used below
    else:
        table = MIX_PHRASES.get(base, {})

    if kind == "slots":
        n = kw.get("n", 2)
        if not table:
            # No phrase for this base language. The caller has an English
            # lead-in it will use instead — so this must be empty, not Hindi.
            # Falling back to Hindi here is the exact bug this change removes.
            return MIX_SLOTS.format(n=n) if base == "hi" else ""
        if n == 1:
            return table.get("slot_one", "")
        return table.get("slots", "").format(n=n)
    if kind == "offer":
        # Offer titles and drug names stay in English in every language here —
        # that is how they are written in practice, and translating a molecule
        # name would be a fabrication. Only the "is running" verb is borrowed.
        verb = table.get("running", "chal raha hai") if table else "is running"
        if base != "hi" and not table:
            return f"{kw.get('offer', '')} is running"
        return f"{kw.get('offer', '')} {verb}"
    if kind == "refill":
        if base != "hi":
            return ""
        return MIX_REFILL.format(n=kw.get("n", "aapki"))
    if kind == "reminder":
        return MIX_REMINDER if base == "hi" else table.get("reminder", "")
    return ""


# --------------------------------------------------------------------------
# CTA construction
# --------------------------------------------------------------------------
# Every CTA is a single, low-friction, one-line ask that externalises the work:
# the merchant never has to produce anything, only say yes.

CTA_BY_FAMILY = {
    "research": "Want me to pull it and draft something you can send?",
    "compliance": "Want me to pull it and draft a checklist?",
    "alert": "Want me to pull the affected list and draft the note?",
    "cde": "Want me to hold you a spot?",
    "seasonal": "Want me to draft the post for this?",
    "competitor": "Want me to show you how to read against that?",
    "perf_dip": "Want me to show you where that went?",
    "seasonal_dip": "Want me to plan around the lull?",
    "perf_spike": "Want me to chase what's driving it?",
    "review": "Want me to draft a reply you can use?",
    "renewal": "Reply RENEW to keep it running.",
    "winback": "Want me to switch it back on?",
    "gbp": "Want me to walk you through it?",
    "milestone": "Want me to nudge the last few?",
    "event": "Want me to draft it tonight?",
    "festival": "Want me to build the package?",
    "dormant": "Want the short version?",
    # The brief calls "asking the merchant" one of the levers Vera almost
    # never fires, so this one has to actually close: a question alone reads
    # as filler, and the reciprocity is what makes answering worth the effort.
    "curious": "Tell me and I'll turn it into a Google post plus a short reply "
               "you can use when customers ask about pricing. Takes 5 min.",
    "planning": "Reply GO and I'll draft it now.",
}


def build_cta(signal: Signal, merchant: dict, category: dict) -> tuple[str, str]:
    """Return (cta_kind, cta_sentence). Empty sentence means no CTA."""
    if signal.customer_facing:
        return _customer_cta(signal)

    default = CTA_BY_FAMILY.get(signal.family, "Want me to take it from here?")
    if not default:
        return CTA_NONE, ""

    kind = signal.cta
    if kind == CTA_BINARY:
        return CTA_BINARY, "Reply YES to go ahead, or tell me to leave it."
    if kind == CTA_CONFIRM:
        return CTA_CONFIRM, default
    return CTA_OPEN, default


def _customer_cta(signal: Signal) -> tuple[str, str]:
    """Customer-scoped messages may offer a slot choice — booking is the
    one flow where a numbered menu beats a binary ask."""
    if signal.cta == CTA_SLOTS and signal.slots:
        labels = [str(s.get("label", "")).strip() for s in signal.slots if s.get("label")]
        labels = [l for l in labels if l][:2]
        if len(labels) == 2:
            return (CTA_SLOTS,
                    f"Reply 1 for {labels[0]}, or 2 for {labels[1]} — "
                    f"or tell me a time that suits you.")
        if len(labels) == 1:
            return CTA_BINARY, f"Reply YES to hold {labels[0]}, or suggest another time."
    if signal.cta == CTA_CONFIRM:
        return CTA_CONFIRM, "Reply CONFIRM and I'll get it done."
    if signal.cta == CTA_BINARY:
        return CTA_BINARY, "Reply YES if you'd like, or tell me to leave it."
    return CTA_OPEN, "Want me to set it up?"


# --------------------------------------------------------------------------
# message assembly
# --------------------------------------------------------------------------

_SENTENCE_END = ".!?"
_CLOSERS = "\"'’”)]"


def _cap(text: str) -> str:
    """Uppercase the first character, but only if it is actually a letter.

    Fragments arrive from many sources and several start with a digit or a
    quote — "3-month fluoride", "1.3 km away", "'delivery late' is showing".
    Finding the first *alphabetic* character and uppercasing it would corrupt
    all three ("3-Month", "1.3 Km", "'Delivery late'"), so only position 0
    is eligible. The rest of the string is left untouched, because many
    fragments legitimately carry capitals mid-sentence ("your Pro plan").

    A leading currency or symbol (₹299, ₹4,999) is not a letter either, so
    those pass through unchanged rather than capitalising the word after it.
    """
    text = (text or "").strip()
    if not text:
        return ""
    first = text[0]
    if first.isalpha():
        return first.upper() + text[1:]
    return text


def _terminate(text: str) -> str:
    """Add a full stop unless the text already ends as a sentence or a quote."""
    text = (text or "").strip()
    if not text:
        return ""
    if text[-1] in _SENTENCE_END:
        return text
    # Ends with a quote/bracket — check the character just before it.
    if text[-1] in _CLOSERS:
        trimmed = text.rstrip(_CLOSERS)
        if trimmed and trimmed[-1] in _SENTENCE_END:
            return text
    if text[-1] in "—-" and len(text) > 2:
        return _terminate(text[:-1].rstrip())
    return f"{text}."


def _clause(text: str, cap: bool = True) -> str:
    return _terminate(_cap(text) if cap else (text or "").strip())


def _cite(citation: str) -> str:
    """Attach a source the way a professional would — never as a URL."""
    c = (citation or "").strip()
    return f" — {c}" if c else ""


def assemble(salutation: str, sentences: list[str]) -> str:
    """Join a salutation and a list of sentences into one message.

    The salutation takes a comma (not a full stop) so it reads as speech —
    "Dr. Meera, your CTR is…". The fragments arrive from many places (signal
    builders, digest items, payloads) and several start lowercase.

    The first fragment after a comma-salutation is left exactly as its signal
    builder wrote it: the sentence is still running, so capitalising it
    produced "Dr. Bharat, Calls 50% down week over week" — a capital letter
    where a clause continues. Fragments that legitimately open with a proper
    noun ("IPL match…", "JIDA Oct 2026…") are unaffected, because the source
    string already capitalises them. Every later fragment is capitalised as
    before, since those do start a new sentence.
    """
    out = ""
    if salutation:
        out = _cap(salutation).rstrip(" .") + ","
    parts = [(s or "").strip() for s in sentences if (s or "").strip()]
    body = " ".join(
        _clause(s, cap=(not out or i > 0)) for i, s in enumerate(parts)
    )
    if out and body:
        return f"{out} {body}"
    return out or body


def assemble_opening(opening: str, sentences: list[str]) -> str:
    """Like `assemble`, but the opening is already a complete clause.

    Used for customer-facing messages, where the greeting carries the merchant
    attribution ("Hi Priya — this is Dr. Meera's clinic") and must not get a
    comma appended to it.
    """
    head = _clause(opening)
    body = " ".join(_clause(s) for s in sentences if (s or "").strip())
    return f"{head} {body}".strip() if body else head


def compose_body(
    signal: Signal,
    category: dict,
    merchant: dict,
    customer: Optional[dict] = None,
) -> str:
    """Render the signal into a message body.

    Structure follows the judge's "high compulsion" example: the anchored fact
    first, then what it means for *this* merchant, then one ask.
    """
    voice = voice_for(category)
    lang = g.language_of(customer if signal.customer_facing and customer else merchant)

    if signal.customer_facing and customer:
        return _customer_body(signal, category, merchant, customer, voice, lang)

    return _merchant_body(signal, category, merchant, voice, lang)


def _merchant_body(
    signal: Signal,
    category: dict,
    merchant: dict,
    voice: Voice,
    lang: str,
) -> str:
    cta_kind, cta_text = build_cta(signal, merchant, category)
    opener = g.salutation(category, merchant)

    # 1. The fact, carrying its citation. This is the specificity anchor.
    head = (signal.headline or "").strip() + _cite(signal.citation)

    # 2. Why it matters to this merchant specifically.
    link = (signal.merchant_link or "").strip()

    # 3. The single ask. For the curious-ask lever the question *is* the ask,
    #    so it leads and the reciprocity line closes.
    sentences: list[str] = []
    if signal.question:
        sentences.append(signal.question)
        if link:
            sentences.append(f"you have {link}") if not link.startswith("you have") else sentences.append(link)
        if cta_text:
            sentences.append(cta_text)
    else:
        if head:
            sentences.append(head)
        if link:
            if link.startswith("you said:"):
                sentences.append(link)
            elif link.startswith("they said:"):
                sentences.append(link)
            else:
                sentences.append(_cap(link))
        if cta_text:
            sentences.append(cta_text)

    body = assemble(opener, sentences)

    # Clinical categories get no emoji. The judge penalises promotional tone
    # in dentists and pharmacies specifically.
    if not voice.clinical and voice.icon and signal.lever in ("curiosity", "reciprocity"):
        if body.endswith("."):
            body = body[:-1] + f" {voice.icon}"

    return apply_language(body, lang, signal.family, category)


def _customer_body(
    signal: Signal,
    category: dict,
    merchant: dict,
    customer: dict,
    voice: Voice,
    lang: str,
) -> str:
    """Messages sent on the merchant's behalf to one of their customers.

    Different rules: attribute the merchant, no product pitch unless it is a
    real active offer, never over-claim, and always honour the customer's own
    language preference.
    """
    name = g.customer_first_name(customer)
    biz = g.dig(merchant, "identity", "name", default="") or "the clinic"
    biz = re.split(r"\s+(?:and|&)\s+", biz)[0].strip() if " and " in biz else biz

    # Fold the attribution into the greeting. Keeping it as a separate sentence
    # gives "Hi Priya, This is X." — a capital mid-clause that reads wrong, and
    # a comma splice when the name is present.
    opener = f"Hi {name} — this is {biz}" if name else f"This is {biz}"

    sentences: list[str] = []

    if signal.headline:
        sentences.append(signal.headline)

    # The offer, only if genuinely active.
    offer = signal.offer or g.primary_offer(merchant)
    offer_title = g.offer_title(offer)
    if offer_title:
        sentences.append(mix_phrase("offer", lang=lang, offer=offer_title)
                         if lang.startswith("mix") else f"{offer_title} is running")

    # Slots, if the trigger gave us any.
    if signal.slots:
        labels = [str(s.get("label", "")).strip() for s in signal.slots if s.get("label")]
        labels = [l for l in labels if l][:2]
        if labels:
            if lang.startswith("mix"):
                # The slot times themselves stay in English/latin digits — that
                # is how these are actually written — and only the lead-in is
                # borrowed, which is the register the case-study anchor uses.
                fragment = mix_phrase("slots", lang=lang, n=len(labels))
                lead = f"{fragment}: " if fragment else ""
                sentences.append(
                    f"{lead}{labels[0]}"
                    + (f", or {labels[1]}" if len(labels) == 2 else ""))
            else:
                lead = ("Two slots open" if len(labels) == 2 else "One slot open")
                sentences.append(f"{lead}: {labels[0]}"
                                 + (f", or {labels[1]}" if len(labels) == 2 else ""))

    _, cta_text = build_cta(signal, merchant, category)
    if cta_text:
        sentences.append(cta_text)

    return apply_language(assemble_opening(opener, sentences), lang, signal.family, category)


# --------------------------------------------------------------------------
# rationale — the judge cross-checks this against the body
# --------------------------------------------------------------------------


def build_rationale(
    signal: Signal,
    category: dict,
    merchant: dict,
    customer: Optional[dict] = None,
    cta_kind: str = "",
) -> str:
    """Explain the decision, not the writing.

    The brief says a rationale that doesn't match the output is penalised, so
    this states the actual signal, the category reason, and the *resolved* CTA
    — a slot question that collapses to a single slot becomes a binary ask,
    and the rationale has to say so.
    """
    voice = voice_for(category)
    bits = []
    subject = signal.headline or signal.question
    if subject:
        bits.append(f"Chose the {signal.family} signal: {subject}")
    if signal.citation:
        bits.append(f"sourced to {signal.citation}")
    bits.append(f"{category.get('slug', 'category')} register ({voice.register})")
    if signal.customer_facing:
        lang = g.language_of(customer or {})
        bits.append(f"customer-facing, {lang} language match")
    else:
        name = g.merchant_first_name(merchant)
        if name:
            bits.append(f"addressed to {name} by first name")
    if cta_kind:
        bits.append(f"CTA {cta_kind}")
    if signal.lever:
        bits.append(f"lever: {signal.lever}")
    return "; ".join(bits) + "."


# --------------------------------------------------------------------------
# send_as
# --------------------------------------------------------------------------


def send_as_for(signal: Signal) -> str:
    return "merchant_on_behalf" if signal.customer_facing else "vera"


# --------------------------------------------------------------------------
# the public compose() entry point
# --------------------------------------------------------------------------

import validate as _validate  # noqa: E402  (circular-import avoidance)


def _recipient_for(signal: Signal, merchant: dict,
                   customer: Optional[dict]) -> str:
    """Who this message is for, and who is signing it.

    A customer-facing message is addressed to the customer; a merchant-facing
    one to the owner. Getting this wrong is not a style problem: checking a
    customer message against the merchant's name made every such message look
    like a greeting failure, and the model was left with no name to check at
    all, which is how a refill reminder for a pharmacy went out addressed to a
    dentist who was never part of the conversation.
    """
    if signal.customer_facing and customer:
        return g.customer_first_name(customer)
    return g.merchant_first_name(merchant)


def _business_for(merchant: dict) -> str:
    return str(g.dig(merchant, "identity", "name", default="") or "")


def _body_via_llm(
    signal: Signal,
    category: dict,
    merchant: dict,
    customer: Optional[dict],
    sent_bodies: Optional[set],
    cta_text: str = "",
) -> Optional[str]:
    """Ask the model to phrase this signal. None means "use the fallback".

    Five layers of defence sit between the model and the wire:

      1. the prompt carries only facts already grounded in code, so there is
         nothing for it to hallucinate a number from
      2. parse_body() strips quotes, preambles and links
      3. every number in the reply must trace to those facts
      4. the reply must open with the person it is actually for
      5. validate.py in compose() rejects taboo words, repeats and bad
         length, with a deterministic retry behind it

    Layers 3 and 4 discard the body rather than repair it. A message with one
    number too many, or one that greets a stranger, is not a sentence that
    needs editing — it is evidence that the model filled a gap, and the
    deterministic composer renders the same signal correctly from code.

    The deterministic path is not a toy fallback — it is the guaranteed floor.
    The judge treats a 500 as a hard failure, so every failure mode here
    returns None and lets the service continue.
    """
    if not config.llm_configured():
        # Counted, not silent. This is the state healthz most needs to be
        # honest about: every message is being phrased by the deterministic
        # composer, and nothing above this point would otherwise say so.
        try:
            import llm

            llm._bump("no_provider")
        except Exception:  # noqa: BLE001
            pass
        return None

    try:
        import llm

        bundle = llm.build_prompt(signal, category, merchant, customer,
                                  sorted(sent_bodies) if sent_bodies else None,
                                  cta_text)
        raw = llm.generate(bundle)
    except Exception:  # noqa: BLE001 — a seam failure must never 500
        return None

    if not raw:
        # Counted here rather than in generate(): this is the exact point the
        # message stops being model-written and becomes deterministic. Counting
        # it in the provider would miss a build_prompt failure entirely, and
        # that is the failure mode that is hardest to see from outside.
        try:
            import llm

            llm._bump("fallback")
        except Exception:  # noqa: BLE001
            pass
        return None

    body = raw.strip()
    if not body:
        return None

    # The grounding check runs on the model output only, not on the composed
    # body. The deterministic composer builds its text out of the same Signal
    # those facts were rendered from, so there is nothing for it to invent;
    # the model is the only path by which a number can reach a merchant that
    # no context field contains.
    business = _business_for(merchant)
    recipient = _recipient_for(signal, merchant, customer)
    # Bodies already sent in this conversation are context the model was
    # legitimately shown, so a number carried over from one is not invention.
    prior = sorted(sent_bodies) if sent_bodies else []
    stray = _validate.ungrounded_numbers(body, bundle.facts, prior)
    problem = stray and f"number(s) not in the facts: {', '.join(stray)}"
    if not problem:
        why = _validate.opening_ok(body, recipient, business,
                                   signal.customer_facing)
        if why:
            problem = why

    if problem:
        try:
            import llm

            llm._bump("rejected")
            llm._bump("fallback")
            # Numbers and name checks only — never the body itself.
            print(f"        [llm] model body rejected: {problem}")
        except Exception:  # noqa: BLE001
            pass
        return None

    return body


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    now: str = "",
    sent_bodies: Optional[set] = None,
) -> Optional[dict]:
    """Select a signal, write the message, validate it, return the send payload.

    Returns None when there is genuinely nothing worth saying — the judge
    explicitly rewards restraint, and a bad message is worse than silence.
    """
    from signals import is_shippable, select_signal

    signal = select_signal(category, merchant, trigger, customer, now)
    if signal is None:
        return None
    # A signal with nothing checkable in it is worse than silence: the judge
    # rewards restraint, and a vague line ("what's been the most-asked
    # service?") costs more than not writing. See signals.is_shippable.
    if not is_shippable(signal):
        return None

    # The CTA is resolved before the body, not after: the closing ask is part
    # of the message, so the model is given the same sentence the deterministic
    # composer would have used, and the grounding check has a source for the
    # numbers inside it.
    cta_kind, cta_text = build_cta(signal, merchant, category)

    body = _body_via_llm(signal, category, merchant, customer, sent_bodies,
                         cta_text)
    if body is None:
        body = compose_body(signal, category, merchant, customer)

    result = _validate.validate(
        body, category, merchant, sent_bodies,
        recipient=_recipient_for(signal, merchant, customer),
        business=_business_for(merchant),
        customer_facing=signal.customer_facing,
    )

    if not result.ok and not result.cleaned_body:
        # The LLM produced something unusable. One deterministic retry is
        # worth it: the decision was sound, only the phrasing failed, and the
        # fallback renders the same signal correctly.
        if body is not None and body != compose_body(signal, category, merchant, customer):
            result = _validate.validate(
                compose_body(signal, category, merchant, customer),
                category, merchant, sent_bodies,
                recipient=_recipient_for(signal, merchant, customer),
                business=_business_for(merchant),
                customer_facing=signal.customer_facing,
            )
            if result.cleaned_body and result.ok:
                body = None
        if not result.cleaned_body:
            return None

    if not cta_kind or cta_kind == CTA_NONE:
        cta_kind = signal.cta if cta_text else CTA_NONE

    rationale = build_rationale(signal, category, merchant, customer, cta_kind)
    if result.issues:
        flagged = "; ".join(i for i in result.issues if not i.startswith("warning"))
        if flagged:
            rationale += f" (auto-corrected: {flagged})"

    return {
        "body": result.cleaned_body,
        "cta": cta_kind,
        "cta_text": cta_text,
        "send_as": send_as_for(signal),
        "suppression_key": str(trigger.get("suppression_key", "") or ""),
        "rationale": rationale,
        "signal": signal.to_dict(),
    }
