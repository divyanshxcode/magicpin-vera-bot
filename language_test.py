"""Language preference is honoured per language, not per "mix".

The dataset ships `hi-en mix`, `kn-en mix`, `ta-en mix` and `te-en mix`.
`language_of` used to collapse every one of them to a bare `mix`, and
`mix_phrase` then emitted Hindi to a Kannada, Tamil or Telugu reader — the
single loudest way to ignore who is on the other end of the message.

The rule these tests pin down: a reader is spoken to in their own language,
and a language we have no phrase for gets English rather than somebody
else's.
"""
import json
from pathlib import Path

import composer
import grounding as g

D = Path("/Users/divyansh-air/Local Disk/Codes/Github Clones/magicpin/dataset")
NOW = "2026-09-27T19:00:00Z"

_passed, _failed = [], []


def check(name, cond, detail=""):
    (_passed if cond else _failed).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if not cond and detail else ""))


def cust(pref):
    return {"customer_id": "c_x", "identity": {"name": "Test Customer", "language_pref": pref},
            "state": "active", "relationship": {"last_visit": "2026-01-12", "visits_total": 6}}


def body_for(pref):
    merch = json.loads((D / "merchants_seed.json").read_text())["merchants"][0]
    cat = json.loads((D / "categories/dentists.json").read_text())
    trig = {"id": "t1", "kind": "recall_due", "scope": "customer",
            "customer_id": "c_x", "urgency": 3,
            "payload": {"service_due": "6_month_cleaning", "last_service_date": "2026-01-12",
                        "due_date": "2026-07-12",
                        "available_slots": [{"label": "Wed 5 Nov, 6pm"},
                                            {"label": "Thu 6 Nov, 5pm"}]}}
    return composer.compose(cat, merch, trig, cust(pref), NOW)["body"]


def main() -> int:
    print("Vera bot — language preference\n" + "=" * 60)

    print("\n--- normalisation keeps the base language ---")
    for pref, want in [("hi-en mix", "mix-hi"), ("kn-en mix", "mix-kn"),
                       ("ta-en mix", "mix-ta"), ("te-en mix", "mix-te"),
                       ("english", "en"), ("hi", "hi"), ("", "en")]:
        check(f"{pref!r} -> {want}", g.language_of(cust(pref)) == want,
              f"got {g.language_of(cust(pref))!r}")
    check("an unknown language falls back to English",
          g.language_of(cust("klingon")) == "en")
    check("a mix with no recognised base stays plain mix",
          g.language_of(cust("xx-en mix")) == "mix")

    print("\n--- each language gets its own words ---")
    bodies = {p: body_for(p) for p in
              ["english", "hi-en mix", "kn-en mix", "ta-en mix", "te-en mix", "ml-en mix"]}
    for p, b in bodies.items():
        print(f"    {p:<12} {b[:96]}...")

    HINDI = ("aapke", "ek slot", "chal raha", "dawaiyaan", "reminder hai", "aapki")
    check("the hi reader gets Hindi", "Aapke liye 2 slots ready hain" in bodies["hi-en mix"])
    check("the kn reader is not handed Hindi",
          not any(w in bodies["kn-en mix"].lower() for w in HINDI),
          bodies["kn-en mix"][:110])
    check("the ta reader is not handed Hindi",
          not any(w in bodies["ta-en mix"].lower() for w in HINDI),
          bodies["ta-en mix"][:110])
    check("the te reader is not handed Hindi",
          not any(w in bodies["te-en mix"].lower() for w in HINDI),
          bodies["te-en mix"][:110])
    check("a language we have no phrase for falls back to English, not Hindi",
          not any(w in bodies["ml-en mix"].lower() for w in HINDI),
          bodies["ml-en mix"][:110])
    check("an English reader gets no mixing at all",
          not any(w in bodies["english"].lower() for w in HINDI))

    print("\n--- structure survives the mixing ---")
    for p, b in bodies.items():
        check(f"{p}: both slots still offered",
              "5 Nov" in b and "6 Nov" in b, b)
        check(f"{p}: the customer is still named", b.startswith("Hi Test"), b[:40])
        check(f"{p}: the business is still attributed", "Dental Clinic" in b, b[:60])
    check("the three translated languages are distinct from one another",
          len({bodies["kn-en mix"], bodies["ta-en mix"], bodies["te-en mix"]}) == 3)

    print("\n" + "=" * 60)
    print(f"  {len(_passed)}/{len(_passed) + len(_failed)} checks passed")
    for f in _failed:
        print(f"    x {f}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
