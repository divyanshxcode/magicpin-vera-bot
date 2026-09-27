# Vera — magicpin AI Challenge

The message engine behind Vera: a FastAPI service exposing the five
judge-facing endpoints, grounded in the four context layers the challenge
defines.

```bash
.venv/bin/python main.py            # or: uvicorn server:app --reload
curl localhost:8080/v1/healthz
```

## Architecture

The design splits along one line: **deciding what to say is deterministic code,
writing the sentence is a model.**

```
POST /v1/context ──▶ store.py          versioned, latest-wins context
                          │
POST /v1/tick ─────▶ tick.py           restraint guards, one action/merchant
                          │
                          ▼
                    signals.py         pick ONE signal, score-ranked
                          │            (decision quality — the judge's
                          │             most heavily weighted dimension)
                          ▼
                    grounding.py       every number, date, offer, citation
                          │            (anti-hallucination — the facts are
                          │             code, so they cannot be invented)
                          ▼
                    composer.py  ──┐
                          ▲       │  Signal -> prose
                          │       ▼
                          │   llm.py  ← the seam
                          │       │
                          │       ▼
                          │   validate.py  URL strip, anti-repeat, taboos
                          ▼
                    { body, cta, send_as, suppression_key, rationale }
```

**Why this split.** The judge scores decision quality, specificity, category
fit, merchant fit, and engagement compulsion. The first two are decided before
a single word is written — which signal leads, and which real number supports
it. A model asked to do that too would be guessing, and would eventually
invent a citation that isn't in the payload. Grounding in code makes
hallucination structurally impossible rather than merely discouraged.

Everything downstream of the `Signal` is prose, and prose is what models are
for. `composer.py` ships a deterministic implementation so the whole pipeline
is testable with no API key; `llm.py` is the drop-in replacement.

## Endpoints

| Method | Path            | Purpose                                        |
|--------|-----------------|------------------------------------------------|
| GET    | `/`             | human-facing index (not judge-facing)          |
| GET    | `/v1/healthz`   | liveness + context counts                      |
| GET    | `/v1/metadata`  | team, model, approach, version                 |
| POST   | `/v1/context`   | push category/merchant/customer/trigger        |
| POST   | `/v1/tick`      | proactive actions, ≤20, with rationale         |
| POST   | `/v1/reply`     | multi-turn reply → `send` / `wait` / `end`     |

## Restraint

`/v1/tick` fires every 5 simulated minutes across a 60-minute window. Sending
whenever a trigger is available means a dozen messages per merchant per run, and
the brief is explicit that *"restraint is rewarded; spam is penalized."* Every
candidate send passes:

- trigger not yet pushed, or expired → skip
- merchant opted out → skip
- one action per merchant per tick
- `suppression_key` already spent (24h window) → skip
- 45-minute per-merchant cooldown
- closed conversation → skip
- customer-scoped trigger with no `CustomerContext` yet → skip
- nothing worth saying → skip, and return `{"actions": []}`

## Reply handling

Three failure modes the brief names, and what we do about each:

| Failure | Response |
|---|---|
| Auto-reply (40–70% of "replies") | detect canned phrasing or verbatim repeat → flag once → wait 4h → end. Does **not** opt the merchant out; only a person can do that. |
| Intent handoff | commitment language switches straight to action with a concrete deliverable. `QUALIFY_RE` exists to assert in tests that no qualifying question leaks out after a commit. |
| Hostility / off-topic | apologise once with an opt-in path, then end. Out-of-scope asks declined honestly, original thread kept alive. |

We also refuse to invent: an unanswered question gets *"let me check and come
back with the real figure"* rather than a fabricated number.

## Tests

```bash
.venv/bin/python smoke_test.py       # 80  endpoint contract + dispatch
.venv/bin/python replay_test.py      # 16  the judge's 3 replay scenarios
.venv/bin/python injection_test.py   # 29  mid-test context injection
.venv/bin/python hardening_test.py   # 52  determinism, latency, malformed input
.venv/bin/python inspect_messages.py # review all 25 composed seed messages
```

177 checks, all passing. Measured p50 for `/v1/tick` is ~1ms against a 30s
budget.

`injection_test.py` exists because the real judge *"injects new facts you
haven't seen — fresh digest items, performance shifts, surprise customer
scopes"* and *"bots that pattern-match the simulator will fail."* It asserts
the bot quotes the new numbers and never the stale ones.

## Deployment

Railway, via `railway.toml` (single replica, healthcheck on `/v1/healthz`).
Set these as service variables — never commit them:

```
TEAM_NAME, TEAM_MEMBERS, CONTACT_EMAIL, BOT_VERSION
LLM_PROVIDER, LLM_API_KEY, LLM_MODEL    # optional; blank = deterministic path
```

Single replica is deliberate: conversation state is in-memory, so a second
replica would hold a different view of the same merchant.
