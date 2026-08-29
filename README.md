# Conversational Shopping Agent — TechJam 2026

A multi-turn shopping agent that finds a hidden target product in a frozen
50,000-item Amazon catalog within ten turns.

**Runs on the Python 3.10+ standard library alone.** No network access, no model
API, no local model, no vector database, no third-party packages. Reported token
usage is zero and model cost is zero.

| | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---|---|---|---|---|
| Official weak BM25 baseline | 0.125 | 0.0680 | 9.81 | 0.119 | 0.1067 |
| **This agent** | **1.000** | **0.7387** | **1.570** | 0.943 | **0.910201** |

Measured with the unmodified official evaluator on the 200 public sessions.
Hit rate is 1.000 in all four scenario types. Runs are deterministic and
repeatable; there is no randomness in the agent.

---

## Quick start

```bash
# Python 3.10+; no dependencies to install
pip install -r requirements.txt        # intentionally a no-op

# obtain the frozen catalog (see docs/participant_kit_README.md)
# and place it at data/catalog.jsonl

python -m evaluator.local_evaluator    # official score -> results.json
```

That single command is the one to run the agent in the official harness.
No environment variables are required or read.

```bash
python -m unittest discover -s tests   # 35 tests
python demo.py                         # walkthrough, one session per scenario
python demo.py --metrics               # evaluator metrics table
```

---

## How it works

### 1. Dual-track intent routing

`respond()` receives a message, a turn number and an anonymised profile. It is
never told `scenario_type`, so intent must be **inferred**. `src/routing.py`
does this every turn from requirement markers ("must be", "I need", "key
requirement"), exploration markers ("still exploring", "open to", "just
looking"), and — weighted above both — whether a *filterable* constraint has
actually landed in dialog state.

| | buying | browsing |
|---|---|---|
| BM25 weight | **0.45** | 0.3 |
| phrase / popularity / profile | 7.0 / 6.0 / 0.05 | 7.0 / 6.0 / 0.05 |
| diversify from turn 1 | no | **yes** |

A customer who has stated a requirement wants precision, so their own
vocabulary carries more weight. A customer who is exploring has given nothing to
filter on, so spreading picks across sellers covers more ground per turn.

State beats wording deliberately: a session that opens "just browsing" and then
states a firm requirement is buying, whatever the opening sounded like.
`Constraint.salvaged` keeps this honest — text scraped out of a sentence no
pattern understood carries conversational scaffolding as well as content, so it
does not count as filterable. Without that distinction almost every reworded
session routes to buying on noise alone.

### 2. Multi-turn state

**Recommend and ask in the same turn.** The protocol permits a message, an
`ask_attribute` and a ranked list in one response, and only a returned list can
convert. Withholding recommendations to ask a question costs a turn and buys
nothing. The agent always returns ten and always attaches a question.

**A non-converting turn is evidence.** If the target had appeared in a scored
top ten the session would have ended, so every product already shown is provably
not the target and is dropped. Ten turns become a sweep of up to a hundred
distinct candidates rather than ten near-identical lists.

That inference has one exception, and getting it wrong cost the entire Intent
Override scenario in an early version. While an override is pending the
evaluator refuses to convert, so a turn can *silently contain* the target.
`DialogState.eliminations_are_valid()` suspends elimination until the override
lands; `apply_override()` then erases revocable slots and re-admits everything
previously ruled out. `demo.py` prints this state on every turn.

### 3. Two redundant retrieval routes

Phrase matching against catalog attribute text is precise but brittle. BM25 over
full product text is imprecise but survives rewording. Scores are blended, with
the phrase bonus scaled by inverse document frequency so a requirement matching
one product counts far more than one matching thousands.

Free-form wording never reaches the template patterns. Those patterns do not
merely miss on it — they *mis-fire*: "I'm looking for a polyester piece" trips
the opening pattern and installs that whole clause as a category, routing the
session into an aisle that does not exist. `looks_templated()` gates this, and
unparsed text falls to lexical salvage plus `resolve_categories()`, which
recovers a product family from a whole sentence by label coverage.

### 4. Runtime orchestration

`src/strategy.py` watches for the failure a fixed pipeline cannot see: the
opening turn routed to the wrong family, so every later turn mines a pool that
cannot contain the target.

| Mode | Trigger | Behaviour |
|---|---|---|
| `focus` | default | Category pool, ranked by constraints |
| `broaden` | two turns without new information, or half the pool shown | Union catalog-wide lexical matches into the pool |
| `diversify` | still stalled after broadening | Cap picks per seller |

Escalation uses only observable state. The agent is never told whether it was
right, so the trigger cannot depend on that.

### Components

| File | Responsibility |
|---|---|
| `agent.py` / `starter/agent.py` | Entry point exporting `Agent` |
| `src/routing.py` | Dual-track intent inference and per-track weights |
| `src/dialog.py` | Conversation state: slots, override, template gating |
| `src/catalog.py` | Catalog load, category/phrase indexes, BM25 |
| `src/shopping_agent.py` | Candidate routing, ranking, clarification policy |
| `src/profile.py` | Profile distillation |
| `src/strategy.py` | Runtime orchestration |
| `demo.py` | Session walkthrough and metrics |
| `heldout_eval.py` | Replays any perturbation set against any session pool |
| `tune.py` | Weight sweep harness |

---

## How this was validated

The public set is 200 sessions of clean templated wording. Two things about it
are unrepresentative of the private 800, and both are measured separately.

### Three session pools

| Pool | n | median reviews | median popularity rank in category |
|---|---|---|---|
| `data/public_set.jsonl` | 200 | 6,846 | 2 |
| `data/matched_sessions.jsonl` | 800 | 1,380 | 7 |
| `data/heldout_sessions.jsonl` | 800 | 12 | 82 |

Real Amazon purchases concentrate on popular products, which is why the public
targets sit at median popularity rank 2. The **matched** pool holds 800 products
disjoint from the public targets, resampled to reproduce that profile — the
closest available proxy for the private set. The third pool samples the catalog
uniformly, making it a deliberate adversarial test of whether the popularity
prior is load-bearing.

| templated wording | TechnicalScore |
|---|---|
| public 200 | 0.9102 |
| matched 800 (unseen products) | 0.8849 |
| long-tail 800 (adversarial) | 0.8729 |

### Seven perturbation styles

Generated blind by a separate model that was never shown the parser, then frozen
to disk so every run is offline and reproducible.

| Style (public 200) | Inherited baseline | This agent |
|---|---|---|
| clean (official wording) | 0.9068 | **0.9102** |
| natural | 0.4967 | **0.8075** |
| terse | 0.4520 | **0.8339** |
| typos | 0.4520 | **0.8380** |
| indirect | 0.4819 | **0.7593** |
| rambling | 0.8815 | **0.8867** |
| categories | 0.8832 | 0.8836 |
| *lossy (information removed)* | *0.8101* | *0.7848* |

On terse and typos the inherited version's hit rate was **0.535** — it missed
the target entirely in nearly half of all sessions, because a message its
patterns did not match produced no category and no constraint, leaving it to
rank all 50,000 products by popularity for the rest of the session.

`lossy` deliberately *removes* information rather than rewording it, so a lower
score is expected there and it is never averaged with the rest.

```bash
python heldout_eval.py --rewrites reports/heldout_natural.json \
                       --dataset data/matched_sessions.jsonl
```

---

## Disclosure: latency, tokens, and cost

| | |
|---|---|
| Network access required | **No.** None, at any point. |
| Model API | None |
| Token usage | **0** prompt, **0** completion |
| Estimated model cost | **$0.00** |
| Per-turn latency | median **8.8 ms**, p95 61 ms, p99 79 ms, max 128 ms |
| One-time index build | 8.2 s at start-up |
| Full 200-session run | 6.7 s |

Measured on the public set; see `reports/latency.json`. The agent reads no
environment variable and opens no socket, so it runs unchanged under the CPU,
memory, timeout and network restrictions the organizer reserves the right to
impose (`docs/submission_rules.md`).

---

## Rejected approaches

Kept here rather than as dead code. A negative result measured properly is still
a result, and each of these is a lever a reader would otherwise assume we missed.

**LLM semantic ranking.** A structured LLM parser for free-form messages reached
0.839 under full rewording — better than the offline path. It was removed
because it requires a live endpoint, and the organizer may score with network
access disabled. Per-turn latency also rose from ~9 ms to ~1.3 s, and with the
endpoint unreachable every message burned the full timeout: a projected 13+
hours for 800 sessions. Reliability beat the ranking gain.

**Suppressing the popularity prior on the buying track.** The original dual-track
design discounted the purchase prior so a stated requirement would not drift
toward best-sellers. It gained on the public set's reworded styles and was
**wrong**: validation on the two independent pools showed the gain is
anti-correlated with target popularity — it helps for long-tail targets and
hurts for popular ones. Since private targets are real purchases, it cost 0.006
on the matched pool. Reverted; see `reports/tier2.md`.

**Cross-category browsing pools.** Implemented as
`CatalogIndex.category_neighbours`. Costs 0.005 on clean wording and changes
nothing under rewording, because reworded sessions never reach the named-label
branch — they resolve through `resolve_categories`, which already pools up to
eight families. Cross-category matching therefore already happens where recall is
actually at risk. Ships disabled.

**A corpus-trained semantic encoder.** An LSA neighbour table over the catalog
produced sensible synonyms (*hiking* → *trekking*; *sneakers* → *trainers*) but
made things slightly worse at every strength tested. Hit rate is already 1.000,
so there is no recall to recover; inside a category-filtered pool synonyms mostly
promote same-family competitors and blur the ranking.

**Raising the profile weight.** Costs score at 0.15, 0.30 and 0.60. The supplied
profile has little to personalise on: `purchase_frequency` is identical across
all 200 sessions and the preference tags are generic.

---

## Limitations, and what we would do next

**MRR is where the remaining score is, and it is nearly information-bound.**
Hit@10 is saturated at 1.000, so the score decomposes as 0.084 of headroom in
MRR and 0.012 in efficiency — 87% of what is left is MRR. The evaluator stops the
turn loop the moment the target enters the top ten, so a target's rank is fixed
at first appearance and can never be improved later. MRR is therefore decided by
turn-1 ranking, on a browsing session where the customer has named only a
category. Rank at conversion is currently 57.5% at rank 1; reaching MRR 0.85
would need ~78%. That is guessing the exact purchased item first out of ~180
category peers from one sentence.

**Weights are tuned on 200 sessions.** The matched-pool result (0.8849) is our
honest estimate for the private set, not the public 0.9102.

**Category routing remains the largest single failure mode.** The perturbations
that cost real score all attack the category label.

**Boundary is the weakest scenario on hard targets** (0.75 hit rate on the
long-tail pool). It is a retrieval problem on the hardest slice — median pool
319, target at popularity rank 103 — not a dialog bug.

**Given more time**, in priority order: a pretrained sentence encoder bundled
locally (the corpus-trained one failed, but a pretrained one is a different
proposition, and vocabulary mismatch is what `indirect` at 0.759 is measuring);
a learned P(purchase | category, profile) prior to replace the hand-set
popularity weight; and a proper held-out tuning split so weights are never
fitted on the set they are reported on.

---

## Reproducing every number in this README

```bash
python -m evaluator.local_evaluator                     # 0.910201
python demo.py --metrics                                # same, with breakdown
python -m unittest discover -s tests                    # 35 tests

# perturbation styles (natural | terse | rambling | typos |
#                      indirect | categories | lossy)
python heldout_eval.py --rewrites reports/heldout_natural.json

# unseen-product pools
python heldout_eval.py --dataset data/matched_sessions.jsonl
python heldout_eval.py --dataset data/heldout_sessions.jsonl
```

`reports/` holds the recorded runs: `tier2.md` (dual-track design and every
ablation), `stocktake.md` (headroom analysis), `matrix.json` (full
style × pool matrix), `latency.json`, and `heldout_manifest.json` (how the
perturbation sets were generated).

---

## Development tools, libraries, and data

- **Tools:** VS Code, git, Python 3.12 (targets 3.10+).
- **Libraries:** Python standard library only — `json`, `math`, `re`,
  `dataclasses`, `collections`, `pathlib`. No frameworks.
- **APIs:** none in the submitted agent. An LLM API was used during development
  to generate the frozen perturbation test sets, and is not required to run,
  score, or reproduce the agent.
- **Data:** the organizer's frozen 50,000-product `Clothing_Shoes_and_Jewelry`
  catalog and 200 public sessions from Amazon Reviews 2023. See
  `DATA_ATTRIBUTION.md`. The two additional session pools are derived from the
  frozen catalog; the catalog itself is never modified.

## Team contributions

*To be completed before submission.*
