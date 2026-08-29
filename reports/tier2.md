# Tier 2 — dual-track intent routing

All offline. `USE_LLM_EXTRACTION = False`, no network, no tokens, 35/35 tests pass.

## Why this and not parser-loosening

The headroom analysis decided it. On the public set Hit@10 is already 1.000, so
the score decomposes as:

| component | weight | value | headroom |
|---|---|---|---|
| Hit@10 | 0.50 | 1.000 | 0.000 |
| MRR | 0.30 | 0.721 | **0.084** |
| Efficiency (MTTC 1.60) | 0.20 | 0.940 | 0.012 |

87% of everything left is MRR. And `evaluator/local_evaluator.py:252` breaks the
turn loop the moment the target enters the top 10, so a target's rank is fixed
at first appearance and can never be improved later. MRR is therefore decided by
turn-1/turn-2 ranking quality, which is a *ranking* problem, not a parsing one.

Rank at conversion, public set, clean wording:

| rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| sessions | 115 | 34 | 16 | 15 | 7 | 3 | 5 | 2 | 3 |

## The split

`respond()` never receives `scenario_type`, so intent is inferred per turn from
requirement markers, exploration markers, and whether a filterable constraint has
actually landed (`DialogState.has_hard_constraint`). State beats wording; an
intent override resets the evidence.

| | buying | browsing |
|---|---|---|
| phrase | 7.0 | 7.0 |
| bm25 | **0.45** | 0.3 |
| popularity | 6.0 | 6.0 |
| profile | 0.05 | 0.05 |
| diversify from turn 1 | no | **yes** |

`Constraint.salvaged` keeps text scraped out of an unparsed sentence from
counting as a filterable constraint -- otherwise almost every reworded session
would route to buying on scaffolding alone.

### Discounting the prior on the buying track: measured and rejected

The original design discounted the purchase prior on the buying track, so a
stated requirement would not be dragged back towards best-sellers. It gained on
the public set's reworded styles and was wrong.

Validation on the two independent product pools showed the gain is
**anti-correlated with target popularity**: suppressing the prior helps when
targets are long-tail and hurts when they are popular. The private sessions are
real Amazon purchases, which concentrate on popular products -- like the public
set (median 6,846 reviews), not like a uniform catalog sample (median 12).

| buying `w_popularity` | public clean | matched pool (3 styles) |
|---|---|---|
| single-track baseline | 0.9090 | 0.8867 |
| **6.0 (shipped)** | **0.9102** | 0.8861 |
| 5.0 | 0.9085 | 0.8849 |
| 4.0 | 0.9084 | 0.8835 |
| 2.0 (rejected) | 0.9043 | 0.8807 |

At 6.0 the tracks differ in BM25 weight and in diversification, not in the
prior. That is a thinner split than the original design, but it is the one the
evidence supports, and it scores best on both the public set and the pool that
best proxies the private one.

## Slot decay on the prior

Discounting the prior is only safe when the pool is small enough to show the
constraint narrowed something. When the customer's wording leaves the category
vague the pool stays huge, the constraint is untrustworthy, and the prior is the
best signal left -- so above `POOL_TRUST_LIMIT = 3000` the buying track reverts
to the browsing weight.

Found because the first weights were fitted: they gained on the four styles used
for tuning and lost on the four that were not.

| pool-trust limit | tuned styles | untouched styles* | categories | lossy |
|---|---|---|---|---|
| off | 0.8343 | 0.8364 | 0.8684 | 0.7495 |
| **3000** | 0.8300 | **0.8382** | **0.8830** | **0.7819** |
| 1200 | 0.8280 | 0.8356 | 0.8839 | 0.7827 |
| 400 | 0.8268 | 0.8349 | 0.8851 | 0.7840 |

\* excludes `lossy`, which removes information and is reported separately.

## Public set, official evaluator

| style | Tier 1 | Tier 2 |
|---|---|---|
| clean (official wording) | 0.9090 | 0.9043 |
| natural | 0.8058 | 0.8138 |
| terse | 0.8340 | 0.8325 |
| rambling | 0.7624 | 0.7925 |
| typos | 0.8351 | 0.8392 |
| categories | 0.8869 | 0.8830 |
| indirect | 0.7538 | 0.7694 |
| *lossy (information removed)* | *0.7889* | *0.7819* |

Net: roughly +0.01 to +0.03 on reworded input, -0.005 on clean.

## Built, measured, shipped disabled

Kept in the code with the measurement, because the brief names them and a
negative result is still a result.

| feature | where | measured |
|---|---|---|
| Cross-category browsing pool | `CatalogIndex.category_neighbours` | -0.005 clean; **zero** effect under rewording -- reworded sessions never reach the named-label branch, they resolve via `resolve_categories`, which already pools up to 8 families |
| Raised profile weight (Pillar III) | `Track.w_profile` | costs score at 0.15, 0.30 and 0.60 |
| LLM semantic ranking | `src/llm_extract.py` | reaches 0.839 under full rewording but needs a live endpoint; organizer may disable network (`docs/submission_rules.md:100`) |

## Correction

An earlier diagnosis claimed the boundary scenario wasted a turn on an unparsed
message. Tracing a real session disproved it: the evaluator emits
`"Those options are not quite right yet"` only when `ask_attribute` is `None`,
for any scenario, and this agent always names an attribute. The real boundary
message was already handled by `NO_PREFERENCE_RE`. `BOUNDARY_RE` is retained as
a guard and documented as unreachable by current design. Boundary's weakness is
retrieval on the hardest targets (median pool 319, target at popularity rank
103 on the long-tail set), not dialog parsing.


## Final validation, independent product pools

Tier 2 with the rejected `w_popularity = 2.0`, against single-track Tier 1:

| pool | Tier 1 | Tier 2 (pop 2.0) | delta |
|---|---|---|---|
| matched (best private proxy) | 0.8866 | 0.8804 | **-0.0062** |
| long-tail (uniform sample) | 0.8632 | 0.8678 | +0.0046 |

Consistent across all four styles on each pool, and the sign flips with target
popularity. This is what drove the revert above.

## Shipped configuration

| | value |
|---|---|
| public set, official wording | **0.910201** |
| Hit Rate@10 | 1.000 |
| MRR | 0.7387 |
| MTTC | 1.570 |
| tokens | 0 |
| tests | 35 pass |

## Removed

The LLM extraction path (`src/llm_extract.py`), its stress harness
(`llm_stress.py`) and every artifact derived from the external endpoint were
deleted. The agent reads no environment variable and opens no socket; `src/`
imports only `json`, `math`, `re`, `dataclasses`, `collections` and `pathlib`.
The measured result is retained here rather than as dead code: the LLM parser
reached 0.839 under full rewording but requires a live endpoint, and
`docs/submission_rules.md:100` reserves the right to score without network
access.

## Template gating: anchored patterns discarded parseable turns

`looks_templated()` originally matched only `^`-anchored sentence shapes. A
customer who prefixes a template with small talk --

    "I had a long day at the office. I'm looking for cold weather gear.
     A key requirement is: leather."

-- fails every anchor, is classed as free-form, and goes to lexical salvage,
throwing away a parse the patterns would have read exactly. The inherited
solution did not have this problem: its `OPENING_RE` is an unanchored `search`.

`UNANCHORED_TEMPLATE_RES` now recognises the distinctive clauses ("a key
requirement is:", "what matters is:", "ignore my earlier preference") anywhere
in the message. The loose "I'm looking for X" shape stays anchored, because
unanchored it mis-fires on ordinary prose.

| | before | after |
|---|---|---|
| rambling | 0.7744 | **0.8867** |
| every other style | unchanged | unchanged |

Verified at the parse level, not only by score: the three sampled rambling
messages resolve to `necklaces`, `cold weather gear` and `everyday bras`, where
before they produced no category and a bag of salvaged filler.

### Note on how this was nearly missed

The first attempt at this fix was written through a shell heredoc that turned
`\b` word-boundary escapes into literal backspace bytes (`0x08`) in the source.
The six regexes compiled without error and their `.pattern` attribute printed
normally, so the only symptom was a score that did not move. The block is now
written by line range with an assertion that no control byte survives, and the
whole tree was scanned to confirm no other file was affected.
