# Conversational Shopping Agent — TechJam 2026 Track 4

A multi-turn shopping agent that locates a hidden target product in a frozen
50,000-item Amazon catalog. Pure Python standard library: no network access, no
model API, no vector database.

## Results

Measured with the unmodified official evaluator on the 200 public sessions.

| | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---|---|---|---|---|
| Official weak baseline | 0.125 | 0.0680 | 9.81 | 0.119 | **0.1067** |
| This agent | **1.000** | **0.7258** | **1.550** | 0.945 | **0.9068** |

Per scenario, hit rate is 1.000 across Buying, Browsing, Intent Override and
Boundary. Mean turns to conversion is 1.25 for Browsing, 1.46 for Buying and
3.70 for Intent Override — the last is near the protocol floor, since the
evaluator suppresses conversion until the override arrives on turn 3 or 4.

Reported token usage is zero and there is no per-turn network call, so latency
is bounded by local scoring. Index construction is a one-time cost at start-up.

## How it works

Three ideas carry most of the result.

**1. Recommend and ask on the same turn.** The protocol permits a message, an
`ask_attribute`, and a ranked list in one response, and only a returned list can
convert. Withholding recommendations to ask a question therefore costs a turn
and buys nothing. The agent always returns a full ten and always attaches a
question.

**2. A non-converting turn is evidence.** If the target had appeared in a
scored top ten, the session would have ended. So every product already shown is
provably not the target and is removed from consideration. Ten turns become a
sweep of up to a hundred distinct candidates rather than ten near-identical
lists.

This inference has one exception, and getting it wrong cost the entire Intent
Override scenario in an early version. While an override is pending the
evaluator refuses to convert, so a turn can silently contain the target. The
agent detects a pending override from the shape of the opening message and
suspends eliminations until the override lands. `DialogState.eliminations_are_valid()`
owns that rule; three tests pin it.

**3. Two redundant retrieval routes.** Phrase matching against catalog
attribute text is precise but brittle. BM25 over the full product text is
imprecise but survives paraphrase. Scores are blended, with the phrase bonus
scaled by inverse document frequency so a requirement matching one product
counts far more than one matching thousands.

### Components

| File | Responsibility |
|---|---|
| `agent.py` | Submission entry point; exports `Agent` |
| `src/catalog.py` | Catalog loading, category index, phrase index, BM25 |
| `src/dialog.py` | Conversation state: constraint accumulation, override, boundary |
| `src/shopping_agent.py` | Candidate routing, ranking, clarification policy |
| `src/profile.py` | Profile distillation: tags, rating style, question order |
| `src/strategy.py` | Runtime orchestration: stall detection and mode switching |
| `tune.py` | Weight sweep harness (builds the index once, re-scores in process) |
| `robustness.py` | Perturbation harness for paraphrase and category stress tests |
| `build_semantic.py` | Trains the LSA neighbour table (disabled; see below) |

### Routing and ranking

The opening turn names the product family, which selects a category-restricted
candidate pool; if the label does not match exactly, the agent falls back to
Jaccard overlap on category tokens, then to catalog-wide BM25. Each candidate
scores as a weighted blend of phrase agreement, BM25 similarity, a popularity
prior, and the anonymised preference profile. Constraints carry weights by how
firmly they were asserted: stated requirements and post-override intent at 3.0,
volunteered answers at 2.0, revocable background preferences at 0.8.

### Personalized context distillation

`src/profile.py` reduces the anonymised aggregate profile to things the ranker
can act on: preference tags expand from abstract concerns into catalog
vocabulary (*warmth* becomes insulated, thermal, fleece), `rating_style`
modulates how far the popularity prior is trusted, and the tag order sets which
attribute to ask about once open questions stop paying.

Measured honestly, this earns close to nothing on this benchmark, and the
section below says why.

### Runtime orchestration

`src/strategy.py` watches for the failure a fixed pipeline cannot see: the
opening turn routed to the wrong product family, so every subsequent turn mines
a pool that cannot contain the target. It escalates through three modes.

| Mode | Trigger | Behaviour |
|---|---|---|
| `focus` | default | Category pool, ranked by constraints |
| `broaden` | two turns without new information, or half the pool already shown | Union the best catalog-wide lexical matches into the pool |
| `diversify` | still stalled after broadening | Cap picks per seller so each turn covers more ground |

Escalation uses only observable state — turns elapsed, whether new information
arrived, how much of the pool is spent. The agent is never told whether it was
right, so the trigger cannot depend on that. Transitions are recorded on the
orchestrator for demo output.

This is what recovers misrouted sessions: hit rate under a missing category
word goes from 0.960 to 0.995, and score from 0.862 to 0.893, with no cost to
the clean set.

### Clarification policy

The agent asks an open question while open questions still yield new
information, then switches to whichever concrete attribute splits the remaining
pool most evenly. Declined attributes are recorded and never re-asked. The
rationale is information gain: with no hypothesis about which attribute
discriminates, letting the customer volunteer what they consider important
dominates guessing an attribute they may not care about.

## Ablation

Each row removes one component from the full system, on the same 200 sessions.

| Variant | Hit Rate@10 | MRR | MTTC | Score | Delta |
|---|---|---|---|---|---|
| Full system | 1.000 | 0.7238 | 1.535 | 0.9064 | — |
| No phrase matching | 1.000 | 0.6533 | 1.650 | 0.8830 | −0.023 |
| No popularity prior | 0.990 | 0.6523 | 2.200 | 0.8667 | −0.040 |
| No clarification questions | 0.995 | 0.6507 | 1.745 | 0.8778 | −0.029 |
| No elimination memory | 1.000 | 0.7137 | 1.535 | 0.9034 | −0.003 |

Two things worth noting. Removing phrase matching entirely still leaves the
system at 0.883, so performance does not depend on verbatim agreement with
catalog text. And elimination memory contributes little here only because
conversion usually happens on turn one or two; it is insurance for the sessions
where early turns miss, and it is what makes the Intent Override scenario work
at all.

## Robustness

The public set is 200 sessions of clean templated language; the private 800 may
be worded differently. `robustness.py` perturbs the customer's wording and
reports the cost of each perturbation. The paraphrase map is hand-built and not
derived from the catalog or any model, so it cannot flatter a corpus-trained
component.

| Perturbation | Hit Rate@10 | MRR | MTTC | Score | Cost |
|---|---|---|---|---|---|
| Clean wording | 1.000 | 0.7258 | 1.550 | 0.9068 | — |
| Requirement vocabulary rewritten | 1.000 | 0.7182 | 1.565 | 0.9042 | −0.003 |
| Category casing changed | 1.000 | 0.7258 | 1.550 | 0.9068 | 0.000 |
| Character dropped from category | 1.000 | 0.7247 | 1.580 | 0.9058 | −0.001 |
| Category words reordered | 1.000 | 0.7205 | 1.555 | 0.9051 | −0.002 |
| Last category word missing | 0.995 | 0.7055 | 1.785 | 0.8934 | −0.013 |
| Category replaced by vague label | 0.980 | 0.7570 | 2.560 | 0.8859 | −0.021 |

Paraphrasing the requirement costs almost nothing, because category routing and
the popularity prior carry the ranking even when constraint vocabulary stops
matching product text.

**Category routing was the real fragility.** An earlier version resolved the
spoken product family to the single best-matching catalog label. On a near-tie
that routed the whole session into the wrong aisle with no recovery path, and a
single missing word cost 0.42 of score:

| | single-best routing | pooled routing | + orchestration |
|---|---|---|---|
| Character dropped from category | 0.8085 | 0.9049 | 0.9058 |
| Last category word missing | 0.4899 | 0.8622 | 0.8934 |
| Category replaced by vague label | 0.8650 | 0.8650 | 0.8859 |
| Clean wording | 0.9064 | 0.9064 | 0.9068 |

The fix pools every plausible family above a confidence floor instead of
committing to one, scores similarity as the better of token overlap and
character-trigram overlap, and falls through to catalog-wide retrieval when no
label clears the floor. Clean-case score is unchanged.

## What did not work

**The aggregate profile carries almost no ranking signal.** Each sub-component
was isolated on the public set:

| Profile component | Clean score |
|---|---|
| Rating-fit term at 0.6 | 0.9044 |
| Rating-fit term at 0.15 | 0.9046 |
| Rating-fit term off | 0.9058 |
| Popularity trust modulated by rating style | 0.9058 |
| Popularity trust flat | 0.9061 |
| Tag vocabulary at weight 0.2 | 0.9051 |
| Tag vocabulary at weight 0.05 | 0.9058 |

The data explains it. `purchase_frequency` is identical in all 200 sessions,
`average_prior_rating` has a median of 5.0 with little spread, and the nine
preference tags are generic — fit, material and comfort each appear in 70–80%
of sessions. There is little to personalize *on*. The rating-fit term measurably
hurt and now defaults to zero; the distillation layer is retained because it
drives question ordering and recommendation explanations without costing
accuracy, and because the private profiles may carry more variance.


**A semantic encoder did not help.** Vocabulary mismatch looked like the
obvious next lever, so a Latent Semantic Analysis layer was trained over the
catalog (`build_semantic.py`) producing a 12,539-term neighbour table with
sensible content — *hiking* → *trekking*, *backpacking*; *sneakers* →
*trainers*; *tee* → *tshirt*. Wired in as weighted query expansion it made
things slightly worse at every strength tested:

| Expansion decay | Clean wording | Paraphrased wording |
|---|---|---|
| 0.00 (off) | 0.9064 | 0.9038 |
| 0.10 | 0.9062 | 0.9038 |
| 0.20 | 0.9060 | 0.9036 |
| 0.45 | 0.9047 | 0.9022 |

The reason is structural. Hit rate is already 1.000, so there is no recall to
recover; within a category-filtered pool, synonyms mostly promote same-family
competitors and blur the ranking. Expansion ships disabled
(`SEMANTIC_EXPANSION = False`) with the code retained so the result is
reproducible. Regenerating the artifact needs numpy and scikit-learn; the agent
itself does not, and runs unchanged when the artifact is absent.

## Limitations

**Popularity prior.** Weighted at 6.0, tuned on 200 sessions. Score plateaus
across 6–18 and split-half validation shows a gap of about 0.005 between folds,
so the setting looks stable rather than fitted to noise. It is still the
component most exposed to a private set whose targets are less popular; 6.0 was
chosen over the marginally better 13.0 for that reason.

**Category routing still dominates.** The two perturbations that cost real
score both attack the category label. A private set whose openings name product
families loosely would land nearer 0.86 than 0.906.

**No pretrained semantic model.** The corpus-trained alternative was tested and
rejected on evidence (above). A pretrained sentence encoder might behave
differently, but on this evidence the expected gain is small and the offline
guarantee is worth more.

**Tuned on 200 sessions.** The public set is small. Expect the private score to
land below 0.906.

## Reproducing

Requires Python 3.10+. No dependencies.

```bash
git clone https://github.com/TechJam2026/techjam-conversational-search.git
cd techjam-conversational-search

# catalog (verify against the published SHA256SUMS)
curl -L -O https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
sha256sum -c SHA256SUMS
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl

# install this agent
cp agent.py starter/agent.py
cp -r src .

python3 -m evaluator.local_evaluator      # -> results.json
python3 -m unittest discover -s tests -v  # 17 tests
python3 robustness.py                     # perturbation stress tests
python3 tune.py                           # optional weight sweep
```

`reports/` holds the recorded runs: `final_results.json`,
`baseline_results.json`, `ablation.json`, and `robustness.json`. Repeat runs are byte-identical —
there is no randomness in the agent.

## Model and cost disclosure

No LLM API, no local model, no network access at inference time. Reported token
usage is zero and model cost is zero. The submission therefore runs unchanged
under the CPU, memory, timeout and network restrictions the organizer reserves
the right to impose. There is no live-credential dependency and so no fallback
path is required.

## Team contributions

*To be completed by the team before submission.*
