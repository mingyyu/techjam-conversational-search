# More Features: From Structural Matching to Transcript-Likelihood Planning

## Purpose

This is an implementation handoff for improving the current `0.974950` public
TechnicalScore without cloning FangryBirds' implementation. The recommended
direction uses the same useful observation—that the public protocol exposes
structured evidence—but develops it into a different algorithm:

> Maintain a calibrated posterior over possible target products from the whole
> conversation, group products that are observationally indistinguishable, and
> jointly choose the next question and recommendation width to maximize expected
> competition utility.

The semicolon parser repair and catalog-derived intent fingerprints are enabling
foundations. The main innovation should be the transcript-likelihood model,
equivalence-class reasoning, and finite-horizon action planner.

Implement this incrementally. Keep every feature independently switchable until
it passes the validation gates below. Do not modify the official evaluator or
catalog, and do not special-case sample IDs or target ASINs.

## Current baseline and measured opportunity

The checked-in agent currently has:

| Evaluation pool | TechnicalScore | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Public 200 | `0.974950` | `1.000000` | `0.985833` | `2.04000` |
| Matched 800 | `0.957786` | `0.988750` | `0.965871` | `2.31750` |
| Long-tail 800 | `0.925450` | `0.967500` | `0.924750` | `2.78625` |

The public result has only four rank slips, all on turn 4:

| Session | Scenario | Target rank |
|---|---|---:|
| `public_0020` | Buying | 4 |
| `public_0076` | Browsing | 6 |
| `public_0083` | Buying | 4 |
| `public_0099` | Browsing | 2 |

Moving only those four targets to rank 1 at their existing turns would produce
approximately `0.979200`. There is additional timing opportunity in sessions
where better inference can identify the target one turn sooner.

The following read-only runtime prototypes were tested; these changes are not
yet present in the repository:

| Prototype | Public | Matched 800 | Long-tail 800 |
|---|---:|---:|---:|
| Current agent | `0.974950` | `0.957786` | `0.925450` |
| Semicolon-safe reveal parsing | `0.975100` | `0.959086` | `0.928775` |
| Exact-card evidence tier | `0.978125` | `0.961906` | `0.934789` |
| Both foundations together | **`0.978225`** | Not measured together | Not measured together |

The combined public prototype retained HitRate@10 `1.0`, raised MRR to
`0.99375`, and reduced MTTC to `1.995`. Its remaining rank slips were
`public_0020` at rank 4 and `public_0076` at rank 2. If a clone-aware policy made
both rank 1 without changing their turns, the score would be approximately
`0.980100`.

Treat these numbers as directional evidence, not a hidden-leaderboard guarantee.
The two 800-session pools contain unseen products, but they still use the same
simulator family.

## What to learn from FangryBirds—and what not to copy

FangryBirds reconstructs the evaluator-style intent card for each catalog item,
matches the observed constraints against those cards, tracks reply order and
shown products, and uses a metric-aware gate to vary recommendation width. Its
reported result is `0.979100`.

The useful lesson is that conversation events carry more information than a bag
of words. The current agent indexes all feature/detail phrases and combines
phrase, BM25, profile, and popularity signals additively. Consequently, lexical
length effects or popularity can outweigh an exact structural explanation of the
transcript.

Do not reproduce FangryBirds' ranking tuple or gating code. In particular:

- Do not retain a preference after an explicit override. Our override MRR is
  `1.0`; FangryBirds' reported override MRR is `0.966667`.
- Do not add dense embeddings merely because they sound more sophisticated.
  FangryBirds' embedding and FTS ablations regressed relative to its simpler
  popularity tie-break.
- Do not globally widen turns 1–3. Fixed width 2 and width 3 experiments scored
  only `0.955075` and `0.942200` respectively.
- Do not use another static score-margin confidence threshold. That approach
  already regressed in this repository because raw score margins are not
  calibrated probabilities.
- Do not import FangryBirds source. Use its concepts as cited prior art and make
  a clean-room implementation from the public competition protocol and
  participant-visible catalog fields.

Reference for attribution only:
<https://github.com/rayyngg/FangryBirds-tiktok-tech-jam>

## Proposed original design

### 1. Represent observations, not just strings

Add an immutable conversation-event representation. A useful shape is:

```text
Observation
  turn: int
  kind: opening | reveal | exhausted | boundary_deferral | override | free_text
  asked_attribute: str | None
  raw_text: str
  interpretations: [(constraint_tuple, probability)]
  template_confidence: float
  revokes_prior_soft_evidence: bool
```

The agent must remember the attribute it asked on the preceding turn. A reveal
is not merely the text `cotton`; it is evidence that asking attribute `other` or
`material` at a particular point produced that reply. Likewise, an
`additional preference` exhaustion reply is negative evidence about a product's
remaining card fields, whereas the one-time Boundary deferral is neutral.

On override:

- remove revocable pre-override evidence;
- preserve genuinely hard constraints;
- clear eliminations made under the obsolete intent;
- start a new evidence epoch so old reply-order observations cannot contaminate
  the replacement intent; and
- retain the current robust behavior rather than FangryBirds' old-preference
  shortcut.

Suggested integration:

- Extend `DialogState` in `src/dialog.py` with `observations`,
  `last_asked_attribute`, and an `evidence_epoch`.
- Set `last_asked_attribute` after `_choose_question()` in
  `src/shopping_agent.py`.
- Keep the current constraints list for backward-compatible lexical fallback.

### 2. Repair semicolon parsing with an interpretation lattice

The current disclosure branch in `src/dialog.py` calls `.split(";")` on every
semicolon. A single catalog feature may itself contain semicolons, so this can
turn one valid constraint into multiple invalid fragments.

The simulator reveals at most two constraints in one reply. For a disclosure
payload containing semicolons, enumerate:

1. the entire payload as one constraint; and
2. every interpretation formed by splitting at exactly one semicolon.

Score each interpretation using support from the current candidate pool:

- exact normalized intent-fingerprint support;
- exact general phrase-index support;
- predicted position/order support;
- number of unsupported fragments; and
- a small complexity prior favoring fewer pieces when support is tied.

Phase 1 may select the best-supported interpretation deterministically. The
posterior model should improve this by retaining the top few interpretations and
marginalizing over them:

```text
P(event | product) = sum over interpretations i of
                     P(i | message, candidate pool) * P(event_i | product)
```

Safety requirements:

- Bound message length and number of candidate split points.
- Fall back to the existing behavior when no interpretation has catalog support.
- Preserve the fuzz-test guarantee that malformed or delimiter-heavy input
  cannot raise or stall.
- Replace the current unit test that assumes every semicolon is a field boundary
  with cases for one semicolon-bearing field, two fields, multiple internal
  semicolons, empty fragments, and unsupported free text.

### 3. Build a clean-room `IntentFingerprint`

Create an independent participant-side representation from visible product
metadata. Do not import `evaluator.local_evaluator` from runtime code.

At minimum, precompute for every product:

```text
IntentFingerprint
  category_label: str
  ordered_constraints: tuple[str, ...]       # at most four protocol fields
  attribute_classes: tuple[str, ...]
  normalized_constraint_set: frozenset[str]
  reply_by_attribute: mapping[str, tuple[str, ...]]
```

Match the public protocol's observable construction rules: normalized
feature/detail values, material/color signals, budget, stable ordering,
deduplication, and truncation. This is model-based inference from disclosed
rules, not target-label memorization.

Implementation notes:

- Put the clean-room builder in a new `src/evidence.py` or
  `src/intent_model.py`, not in the evaluator.
- Have `CatalogIndex` precompute fingerprints and inverted maps such as
  `constraint -> product positions` and `fingerprint -> product positions`.
- Keep initialization deterministic and measure the extra memory/startup cost.
- Add a development-only parity test against the official evaluator's public
  card builder for all 50,000 catalog products. Runtime code must remain
  independent of the evaluator package.
- Treat exact structural evidence as dominant but never as the only retrieval
  route. If template confidence is low, retain the current BM25/phrase path.

### 4. Replace exact-count ranking with counterfactual transcript likelihood

This is the first primary innovation.

For every candidate product, simulate what replies its fingerprint would have
produced under the questions actually asked. Score how well that counterfactual
transcript explains the observed transcript.

A practical model is:

```text
log weight(product) =
    log prior(product)
  + category log-likelihood
  + sum(event log-likelihoods)
  + gated residual lexical likelihood
```

Then normalize with log-sum-exp to obtain a posterior over the candidate pool.

Use an explicit prior rather than allowing popularity to swamp evidence:

```text
P0(product) = eta * Uniform(product)
            + (1 - eta) * SmoothedPopularity(product, profile)
```

The uniform component keeps long-tail products alive. The popularity exponent
and mixture weight must be chosen on frozen development data and validated over
a neighborhood, not fitted to four public failures.

Suggested event likelihood order:

| Observed relationship | Relative evidence |
|---|---|
| Exact predicted reply, including order | Very strong |
| Same exact fields but different grouping/order | Strong |
| Exact field presence elsewhere in fingerprint | Moderate |
| Supported substring or high lexical similarity | Weak |
| Predicted exhaustion matches observed exhaustion | Strong negative/positive evidence as appropriate |
| Exact-template contradiction | Strong penalty |
| Low-confidence/free-text contradiction | Soft penalty only |
| Boundary deferral | Neutral |

Important refinements beyond exact-card counting:

- Compare the whole sequence of asks and replies, not independent field matches.
- Strip spans already explained by exact structural matches before computing
  residual BM25, preventing the same clue from being counted twice.
- Make evidence confidence continuous. Exact official templates can strongly
  influence the posterior; natural paraphrases should blend smoothly toward the
  current ranker.
- Use floors rather than hard zeroes except for catalog-invalid or safely
  eliminated products. Parser uncertainty should not create false exclusions.
- Make category a dominant likelihood term, not a permanent eligibility filter.
  Union exact-bucket, fuzzy-category, structural, lexical, and popularity rescue
  candidates, then let evidence order them.
- Keep stable deterministic tie-breaking by popularity and ASIN.

The current score can remain available as a fallback or low-confidence residual:

```text
final evidence = posterior evidence when calibrated
               + residual lexical evidence not already explained
               + current ranker when posterior confidence is low
```

Do not blend two uncalibrated raw score scales. Convert each component to a
bounded likelihood, percentile, or within-pool normalized value first.

### 5. Discover observational equivalence classes

This is the second primary innovation and directly addresses the residual public
failures.

At the current disclosed state, define a product's reply signature as the reply
it would produce for every eligible next question:

```text
signature(product, state) = tuple(
    predicted_reply(product, attribute, already_disclosed)
    for attribute in eligible_attributes
)
```

Products with the same signature are observationally equivalent: another
question cannot distinguish them under the current protocol. Group products by
this signature and track for each class:

- posterior probability mass;
- number of unseen members;
- best current rank and popularity ordering;
- attributes that split it on a later turn; and
- whether the class is exact-template-supported or only lexical.

This provides an interpretable policy signal:

- If the posterior is spread over classes that a question can separate, ask and
  keep the list narrow.
- If most posterior mass is inside one large class that no question can split,
  deliberately page through multiple members because waiting has little
  information value.
- If the top product itself has high posterior mass, return width 1 to protect
  MRR.

This is more principled than either fixed `1/1/1/10` or globally wider lists.

### 6. Jointly plan the question and recommendation width

This is the third primary innovation. FangryBirds evaluates a narrower show-now
versus wait decision. Instead, choose the structured question and list width as
one action because they affect the same next state.

For a hit at turn `t` and rank `r`, the per-session contribution implied by the
official metric is:

```text
R(t, r) = 0.50 + 0.30 / r + 0.20 * (11 - t) / 10
        = 0.50 + 0.30 / r + 0.02 * (11 - t)
```

For action `(attribute a, width k)`, compute a bounded lookahead:

```text
V_t(state) = max over (a, k) of [
    sum over target at displayed rank r <= k of posterior(target) * R(t, r)
  + sum over possible non-hit replies y of P(y) * V_(t+1)(updated state | y)
]
```

The transition must model both effects of an action:

1. If the target is displayed, the session ends at that rank.
2. Otherwise, safely eliminable displayed products are removed and the reply to
   question `a` updates the posterior.

Keep it computationally modest:

- consider widths `{1, 2, 3, 5, 10}`;
- consider only eligible attributes plus `other`;
- aggregate candidates by reply/equivalence class;
- use the top 200–500 products plus a tail-mass bucket;
- start with a two-turn horizon and memoize state signatures;
- use the current policy as the terminal value approximation; and
- log the chosen action and the runner-up value for debugging.

Conservative guards:

- While an override is suspected but has not arrived, preserve the existing
  elimination safety and default to a narrow list.
- After an override, start planning from the replacement intent only.
- With low parser confidence, uncalibrated posterior, excessive tail mass, or an
  out-of-distribution message, fall back to the current `1/1/1/10` schedule.
- Never return more than `top_k`, and preserve the output-contract behavior for
  malformed input.

Start the planner in shadow mode: compute and trace the proposed action while
the current policy still controls output. Enable it only after the counterfactual
values agree with known toy cases and improve all required evaluation pools.

### 7. Make the policy distributionally robust

A policy optimized only against the exact public simulator may be clever but
brittle. Evaluate each candidate policy against a small ensemble of plausible
protocol variants without changing the official evaluator files:

- material/color injection absent or moved;
- hard/soft ordering changed;
- three or five intent fields instead of four;
- alternative normalization/truncation;
- paraphrased reveal and exhaustion templates;
- override arriving on turns 2–6;
- ambiguous category labels; and
- semicolon-rich or partially lossy disclosures.

Choose parameters by robust value rather than maximum public mean, for example:

```text
robust_value = mean(TechnicalScore across variants)
             - lambda * standard_deviation(across variants)
```

Alternatively, maximize a lower confidence bound or the worst non-lossy variant.
The exact choice matters less than requiring the policy to win across a stable
neighborhood of simulator assumptions.

## Suggested code map

| File | Proposed responsibility |
|---|---|
| `src/catalog.py` | Precompute fingerprints, structural inverted indexes, and category rescue candidates |
| `src/dialog.py` | Store observations, ask history, evidence epochs, and override-safe state transitions |
| `src/evidence.py` (new) | Clean-room fingerprint builder, interpretation lattice, reply simulator, likelihoods, posterior update |
| `src/planner.py` (new) | Equivalence classes, action enumeration, bounded lookahead, safe fallback decision |
| `src/shopping_agent.py` | Orchestrate retrieval, posterior ranking, planner, and existing fallback |
| `src/config.py` (optional) | Named feature flags and frozen parameters for clean ablations |
| `tests/test_evidence.py` (new) | Fingerprint, parsing, likelihood, and posterior tests |
| `tests/test_planner.py` (new) | Equivalence and action-value tests on synthetic catalogs |

Keep the submitted path standard-library-only, deterministic, offline, and free
of imports from `evaluator/`.

## Phased implementation plan

### Phase 0: Freeze and instrument

1. Re-run and record current public, matched, and long-tail results.
2. Freeze a hash-based audit subset before choosing any new thresholds.
3. Add optional per-turn traces containing:
   - candidate count;
   - target-independent score components;
   - parser interpretations/confidence;
   - fingerprint/equivalence-class sizes;
   - posterior entropy and top-product probability;
   - predicted reply for each considered question;
   - chosen width/action value; and
   - whether a fallback guard fired.
4. Confirm traces do not change output ordering or latency materially when off.

Exit condition: deterministic reproduction of the current metrics and per-session
outputs.

### Phase 1: Ship the low-risk parser repair

1. Add support-aware semicolon interpretation.
2. Add unit and fuzz tests.
3. Run public, matched, long-tail, and wording-perturbation suites.
4. Keep this change only if the positive delta survives outside public.

Expected directional result: public near `0.975100`, matched near `0.959086`,
and long-tail near `0.928775`.

### Phase 2: Add fingerprints and transcript likelihood

1. Implement and parity-test the fingerprint builder.
2. Store structured observations and reply history.
3. Run posterior ranking in shadow mode beside the existing `_rank()`.
4. Inspect calibration, contradictions, and changed session pairs.
5. Enable it first only for high-confidence template observations and outside a
   pending override.
6. Blend toward the existing ranker for low-confidence/free-text input.

Expected directional result for the simpler structural foundation: public near
`0.978125`, matched near `0.961906`, and long-tail near `0.934789`. The posterior
model should equal or improve those results while being less brittle under
protocol perturbations.

### Phase 3: Add equivalence-aware joint planning

1. Implement reply signatures and equivalence classes.
2. Validate planner decisions on synthetic posterior distributions.
3. Run the planner in shadow mode and compare its actions with the fixed policy.
4. Enable it behind conservative guards.
5. Examine `public_0020` and `public_0076` as diagnostics, not special cases.

The goal is to page through genuinely indistinguishable clones without creating
new rank slips elsewhere. Do not accept a policy merely because it fixes those
two named sessions.

### Phase 4: Robust category mixture and simulator ensemble

1. Replace single-bucket eligibility with a probability mixture over plausible
   categories plus a global lexical rescue pool.
2. Add the simulator-variant harness and robust action selection.
3. Tune only after the formulas and audit split are frozen.

This phase is aimed primarily at private-set and reworded-input robustness, not
at extracting another public-set decimal.

## Required tests

### Unit tests

- Fingerprint parity over all catalog products in a development-only test.
- Stable normalization, deduplication, ordering, and truncation.
- One constraint containing internal semicolons.
- Two revealed constraints, either or both containing internal semicolons.
- Unsupported and malformed disclosures falling back safely.
- Exact reply-order match scoring above unordered membership.
- Exhaustion evidence penalizing candidates with undisclosed matching fields.
- Boundary deferral contributing no negative product evidence.
- Override starting a new evidence epoch and re-admitting prior recommendations.
- A posterior that sums to one and remains nonzero for long-tail candidates.
- Deterministic ordering under exact ties.
- Equivalence-class construction for identical and distinguishable cards.
- Planner chooses width 1 when top-one probability is high.
- Planner widens when a high-mass class cannot be split by any question.
- Planner asks the splitting attribute when expected information is valuable.
- Pending-override and low-confidence states trigger the safe fallback.

### Integration and robustness tests

Run at least:

```text
python -m unittest discover -s tests -v
python -m evaluator.local_evaluator
python heldout_eval.py --dataset data/matched_sessions.jsonl
python heldout_eval.py --dataset data/heldout_sessions.jsonl
```

Then run both public and held-out sessions through the existing `natural`,
`terse`, `rambling`, `typos`, `indirect`, `categories`, and `lossy` rewrite maps.
Treat `lossy` as a stress result rather than a strict ship gate because it
intentionally deletes information.

Report all metrics overall and by:

- scenario;
- target popularity quartile;
- category-pool size;
- parser-confidence band;
- semicolon-bearing versus ordinary fingerprints;
- fingerprint/equivalence-class size (`1`, `2–5`, `6+`); and
- clean versus reworded input.

Also record rank-1 rate, number of misses, paired per-session delta, p95/p99 turn
latency, startup time, and peak memory.

## Acceptance gates

Do not select a feature on public TechnicalScore alone.

Minimum ship gates:

1. No HitRate@10 loss on public or matched pools.
2. Positive paired TechnicalScore delta on both matched and long-tail pools.
3. No regression on at least 80% of the non-lossy wording suites; investigate
   every regression larger than `0.001`.
4. No loss of the current perfect public override MRR unless a larger,
   independently validated gain clearly compensates for it. Prefer no loss.
5. No sample-ID, target-ASIN, public-target-frequency, or ground-truth feature.
6. No network dependency, nondeterministic model call, or evaluator import in
   submitted runtime code.
7. The improvement persists across a neighborhood of parameter values instead
   of one sharp optimum.
8. Startup, memory, and turn latency remain within the documented operational
   envelope; investigate any p95 turn-latency regression above 25%.
9. Existing output-contract and adversarial fuzz tests continue to pass.

Use paired bootstrap confidence intervals for per-session score deltas when
comparing close variants. A small mean gain produced by one lucky public session
is not enough.

## Anti-overfitting and compliance checklist

- Never read `sample_id` inside the agent.
- Never encode or look up known target ASINs.
- Never use public target frequency as a ranking feature.
- Never mutate `data/catalog.jsonl`, official session files, or evaluator code.
- Build fingerprints only from participant-visible catalog fields.
- Keep exact-template logic as a high-confidence evidence channel, not a hard
  filter that disables free-text recovery.
- Derive constants from the metric/protocol where possible. For every empirical
  constant, include a sensitivity sweep.
- Freeze development/audit partitions before tuning.
- Make one conceptual change per ablation and record the commit SHA.
- Keep a rollback flag for each phase until every ship gate passes.
- Cite FangryBirds as inspiration; do not copy its source or present its card
  reconstruction/gate as this project's novel contribution.

## Ablation record template

For every experiment, create a short machine-readable result plus a Markdown
decision note containing:

```text
Hypothesis:
Code paths changed:
Feature flags and parameter derivation:
Public / matched / long-tail metrics:
Scenario and robustness-slice deltas:
Changed sessions and paired score deltas:
New misses or rank>1 conversions:
Parser calibration and posterior Brier/ECE:
Startup / memory / p50 / p95 / p99 latency:
Sensitivity sweep:
Decision: ship | revise | reject
Reason:
```

The decision log matters because this score is already near saturation. A
feature can appear to improve the total while silently trading one expensive miss
for several cheap rank gains.

## Definition of done

The work is complete when:

- the semicolon repair and structured observation model are tested;
- the clean-room fingerprint builder has full-catalog parity in development;
- the transcript posterior is calibrated and has a robust fallback;
- equivalence-aware planning jointly selects question and width;
- public, matched, long-tail, perturbation, scenario, latency, and fuzz reports
  are recorded;
- all acceptance gates pass; and
- the README/report describes the contribution accurately.

A concise way to describe the final innovation is:

> The agent treats conversational shopping as online diagnosis. It evaluates
> which products could have generated the entire observed transcript, maintains
> uncertainty over those hypotheses, and chooses questions and recommendation
> widths by their expected value under the competition metric. Products that
> would answer every possible question identically are recognized as an
> equivalence class and paged directly instead of wasting a clarification turn.

