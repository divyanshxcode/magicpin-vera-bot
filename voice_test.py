"""The category pack's own voice reaches the model, and its facts reach the body.

Two findings from a real judge run drove this.

`_voice_block` read `taboo_words` and `do_not_say` — field names that exist in
no category pack. The dataset uses `vocab_allowed` and `vocab_taboo`, and
`vocab_taboo` is the field the judge weighs when it scores promotional
register. So the model was told the tone and never told what to avoid. Every
category also ships `tone_examples` and `salutation_examples`, which nobody was
reading; those show the voice rather than describe it.

The second is that `_customer_body` rebuilt its sentences from the headline,
the offer and the slots, and never read `signal.merchant_link`. Every merchant
link a customer-facing signal carried was built, ranked, and thrown away.
"""
import json
import logging
from pathlib import Path

import composer
import grounding as g
import llm
import signals as S

D = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")
NOW = "2026-09-27T19:00:00Z"

_passed, _failed = [], []


def check(name, cond, detail=""):
    (_passed if cond else _failed).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if not cond and detail else ""))


def cat(slug):
    return json.loads((D / f"categories/{slug}.json").read_text())


def merchants():
    return {m["merchant_id"]: m for m in
            json.loads((D / "merchants_seed.json").read_text())["merchants"]}


def triggers():
    return {t["id"]: t for t in
            json.loads((D / "triggers_seed.json").read_text())["triggers"]}


def customers():
    return {c["customer_id"]: c for c in
            json.loads((D / "customers_seed.json").read_text())["customers"]}


def main() -> int:
    logging.disable(logging.CRITICAL)
    print("Vera bot — category voice and merchant links\n" + "=" * 60)

    # ---------------------------------------------------------------- voice
    print("\n--- the voice block carries what the pack actually ships ---")
    for slug in ("restaurants", "salons", "dentists", "gyms", "pharmacies"):
        c = cat(slug)
        v = c.get("voice") or {}
        block = llm._voice_block(c)
        check(f"{slug}: taboo list reaches the block",
              any(t in block for t in (v.get("vocab_taboo") or [])[:2]),
              f"missing {[t for t in (v.get('vocab_taboo') or [])[:2]]}")
        check(f"{slug}: the pack's own examples reach the block",
              all(e in block for e in (v.get("tone_examples") or [])[:2]),
              "tone_examples missing")
        check(f"{slug}: the pack's greeting style reaches the block",
              any(s in block for s in (v.get("salutation_examples") or [])[:1]),
              "salutation_examples missing")

    print("\n--- and into the prompt itself ---")
    c = cat("restaurants")
    sig = S.Signal(family="event", headline="DC vs MI", why_now="w",
                   merchant_link="l", cta="open_ended")
    b = llm.build_prompt(sig, c, {"identity": {"name": "X", "owner_first_name": "S"}},
                         None, "cta text")
    check("the voice block is in the user message", "CATEGORY VOICE" in b.user)
    check("the taboo list is in the user message", "words to avoid" in b.user)
    check("the examples are in the user message",
          "examples of this category" in b.user)
    check("a pack with no voice at all produces no block",
          llm._voice_block({"slug": "x"}) == "")
    check("a non-dict voice does not crash", llm._voice_block({"voice": "nope"}) == "")

    # ------------------------------------------------------- merchant links
    print("\n--- customer-facing messages use their merchant link ---")
    mers, trigs, cus = merchants(), triggers(), customers()
    t = trigs["trg_007_bridal_followup_kavya"]
    m = mers[t["merchant_id"]]
    c = cat("salons")

    body_rec = composer.compose(c, m, t, cus[t["customer_id"]], NOW)["body"]
    body_judge = composer.compose(c, m, t, {"customer_id": t["customer_id"]}, NOW)["body"]

    check("the bridal service is named when the record is present",
          "Bridal Trial" in body_rec, body_rec[:110])
    check("the catalogue service is named when it is not",
          "Bridal Trial" in body_judge, body_judge[:110])
    for label, b in (("with record", body_rec), ("judge push", body_judge)):
        check(f"{label}: the orphan programme name is gone",
              "skin prep" not in b.lower() and "skin_prep" not in b.lower(), b[:120])
        check(f"{label}: the countdown still comes from the payload",
              "196 days" in b, b[:90])
        check(f"{label}: the service is named exactly once",
              b.lower().count("bridal trial") == 1, b[:130])
        check(f"{label}: no mid-sentence full stop",
              ". " not in b[:-1] or b.endswith("?"), b[:130])
        check(f"{label}: the business is still attributed",
              "Studio11" in b, b[:70])

    print("\n--- the match-night insight is read, not hardcoded ---")
    ins = S._match_night_insight(cat("restaurants"))
    check("the Saturday penalty is extracted from the digest", "bad" in ins, str(ins)[:120])
    check("the weeknight gain is extracted from the digest", "good" in ins, str(ins)[:120])
    check("the figures come from the pack",
          "12%" in ins.get("bad", "") and "18%" in ins.get("good", ""), str(ins)[:160])
    check("a category with no match evidence returns {}",
          S._match_night_insight(cat("dentists")) == {})
    check("no category returns {}", S._match_night_insight(None) == {})

    t2 = trigs["trg_010_ipl_match_delhi"]
    m2 = mers[t2["merchant_id"]]
    ev = composer.compose(cat("restaurants"), m2, t2, None, NOW)["body"]
    check("a Saturday match uses the Saturday finding",
          "Saturday" in ev and "12%" in ev, ev[:130])
    check("the CTA matches the finding rather than the family",
          "hold Saturday" in ev, ev[-90:])

    same = dict(t2, payload=dict(t2["payload"], is_weeknight=True))
    ev_wd = composer.compose(cat("restaurants"), m2, same, None, NOW)["body"]
    check("a weeknight match uses the other finding",
          "18%" in ev_wd and "Saturday" not in ev_wd.split(";")[0], ev_wd[:130])

    print("\n--- catalogue lookup stays inside its own pack ---")
    check("a salon slug finds the salon catalogue entry",
          g._match_catalogue_title("bridal_trial", cat("salons")) == "Bridal Trial @ ₹999",
          g._match_catalogue_title("bridal_trial", cat("salons")))
    check("a service from another vertical is not matched",
          g._match_catalogue_title("bridal_trial", cat("dentists")) == "")
    check("an unknown service returns nothing rather than a wrong title",
          g._match_catalogue_title("quantum_tuning", cat("salons")) == "")
    check("with no pack the slug is de-underscored, not invented",
          g.customer_service({"relationship": {"services_received": ["hair_spa"]}}) == "hair spa")
    check("a customer with no services yields nothing",
          g.customer_service({"relationship": {}}) == "")

    print("\n" + "=" * 60)
    print(f"  {len(_passed)}/{len(_passed) + len(_failed)} checks passed")
    for f in _failed:
        print(f"    x {f}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
