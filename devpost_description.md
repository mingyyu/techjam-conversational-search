# Conversational Shopping Agent

A multi-turn shopping agent that finds a customer's intended product in a frozen
50,000-item Amazon catalog within ten turns — running entirely on the Python
standard library, with **zero API calls, zero tokens, and zero cost**.

**TechnicalScore 0.974950** on the official evaluator (Hit Rate@10 **1.000**,
MRR 0.9858, MTTC 2.040), against a published baseline of 0.1067 — a 9.1×
improvement. Hit Rate@10 is 1.000 in all four scenario types, and 196 of the
200 conversions land at rank 1.

---

## How it addresses the problem statement

**Dual-track intent routing.** The agent is never told which scenario it is in —
`respond()` receives only a message, a turn number and an anonymised profile — so
intent has to be *inferred*. Every turn it re-reads requirement markers ("must
be", "I need"), exploration markers ("still exploring", "open to"), and, weighted
above both, whether a genuinely filterable constraint has landed in state. A
buying turn leans harder on the customer's own vocabulary; a browsing turn
spreads picks across sellers to cover more ground. State beats wording
deliberately: a session that opens "just browsing" and then states a firm
requirement is buying, whatever the opening sounded like.

**Multi-turn state with override handling.** Slots accumulate across turns and
carry weights by how firmly they were asserted. An intent override erases the
revocable ones and rewrites intent. The subtle part is what happens *before* the
override lands: the protocol suppresses conversion until it arrives, so a turn
can silently contain the target. The agent detects this and suspends its
elimination memory rather than discarding the answer. Getting that one rule right
moved the Intent Override scenario from 0.233 to ~1.000.

**Proactive clarification by information gain.** The agent recommends *and* asks
in the same turn — the protocol allows it, so spending a turn on a question alone
costs a turn and buys nothing. It asks an open question while open questions keep
paying, then switches to whichever attribute splits the remaining candidate pool
closest to 50/50, because that eliminates the most candidates whichever way the
customer answers. Declined attributes are never re-asked.

**Runtime re-orchestration.** A fixed pipeline cannot see its own worst failure:
the opening turn routed to the wrong product family, so every later turn mines a
pool that cannot contain the target. A supervisor watches for the stall using
only observable signals — turns elapsed, whether new information arrived, how
much of the pool is spent — and escalates `focus → broaden → diversify`.

---

## What we are most proud of

**We built the test set that could prove us wrong.**

We are scored on 200 public sessions and graded on 800 private ones that use
**different users and different target products**. Tuning against a 200-session
set with about a dozen knobs is how you produce a number that does not survive
the swap, so most of our effort went into finding out where it would break rather
than into polishing the number we could see.

**The real exposure is product generalisation, so that is what we measured
first.** We built two additional 800-session pools from the same frozen catalog:
one resampled to reproduce the *purchase-popularity profile* of real Amazon
orders, which is the closest available proxy for the private set, and one sampled
uniformly as a deliberately adversarial check on whether our popularity prior was
load-bearing.

| Pool (official wording) | TechnicalScore | Hit@10 |
|---|---|---|
| public 200 (the set we can see) | 0.974950 | 1.000 |
| **matched 800 — unseen products, purchase-like popularity** | **0.957786** | 0.989 |
| long-tail 800 — uniform sample, adversarial | 0.925450 | 0.968 |

**0.9578 is our honest estimate for the private set, not the public 0.9749.**
And if the private targets turned out to be long-tail after all — which the
specification makes unlikely, since it anchors them on real purchase records —
we would give up a further 0.032, with Hit@10 still at 0.968. There is no cliff
anywhere in that range.

**Then we hardened the parts a benchmark cannot see.** The specification reserves
the right to add natural-language paraphrasing, and any agent that would survive
contact with a real shopper has to read free text anyway. So we generated **seven
perturbation styles** — natural, terse, rambling, typos, indirect, renamed
categories, and information-loss — using a separate model that was deliberately
never shown our parser, so the test could not be tuned against.

That harness found things the score alone never would:

- The inherited version's hit rate collapsed to **0.535** on lightly reworded
  input — it missed the target outright in nearly half of all sessions. Clean
  score said 0.907.
- A ranking change that gained on every style we had tuned on turned out to be
  *anti-correlated with target popularity*, and would have cost us on the private
  set. We reverted it.
- A template-matching bug that silently discarded fully parseable turns.

| Style (public 200) | Inherited | Ours |
|---|---|---|
| official wording | 0.9068 | **0.974950** |
| rambling | 0.8815 | **0.936576** |
| renamed categories | 0.8832 | **0.931116** |
| terse | 0.4520 | **0.890260** |
| typos | 0.4520 | **0.882156** |
| natural | 0.4967 | **0.841981** |
| indirect | 0.4819 | **0.802883** |

We also perturbed **every fitted constant at once** by a uniform random factor, to
test whether our parameters sat on a peak or a plateau. At ±25% the worst of eight
draws costs 0.003 and Hit@10 stays at 1.000; at ±50% one draw scores *above* the
shipped configuration. The constants are not load-bearing — the architecture is.

---

## What we learned

**We cut the LLM on purpose, and measured what it cost.** An LLM parser for
free-form messages scored **0.839** under full rewording — better than our
offline path. We removed it anyway. It needs a live endpoint, and the organizer
reserves the right to score with network access disabled; per-turn latency went
from ~9 ms to ~1.3 s, and with the endpoint unreachable every message burned the
full timeout — a projected 13+ hours for 800 sessions. A solution that cannot run
in the grading environment scores zero, whatever it scores in ours.

**How many products to return is a decision, not a formality — and it was worth
more than any weight we tuned.** The evaluator stops the turn loop the instant
the target enters the top ten, so a product's rank is fixed at first appearance
and can never be improved later. That makes a low-ranked guess the opposite of a
hedge: it locks the rank in. Priced against one session in two hundred, an extra
turn costs 0.0001 of score, lifting a session from rank 3 to rank 1 gains
0.0010, and losing a hit costs 0.0025. So while the customer still has something
to disclose the agent returns only its single best candidate and spends the turn
asking; once they run dry it returns the full ten and sweeps. Public 0.9095 →
**0.9749** and matched-800 0.8901 → **0.9578**, Hit@10 unchanged at 1.000, and
every one of the sixteen pool-by-wording combinations we measured improved.

The commitment window is three turns because that is the *simulator's* schedule,
not a number we fitted: an intent card holds at most four constraints and a
reply releases two, so nothing new arrives after turn 3. Past that the curve
turns over on Hit@10, which is exactly where the pricing says it should.

**Most of the remaining headroom is genuinely unreachable.** With Hit@10 at
1.000 and 196 of 200 conversions at rank 1, 0.019 of score is left and MTTC
cannot fall below 1.390 — an intent-override session is forbidden from
converting before its override lands on turn 3 or 4. Knowing that stopped us
burning time on a wall.

**Four ideas we built, measured, and switched off:** cross-category browsing
pools, a corpus-trained LSA semantic encoder, a raised personalisation weight,
and a confidence gate on the commitment policy (committing only when the top
candidate's score margin cleared a threshold — worse at every threshold tested). All are documented with their numbers. A negative result measured
properly is still a result.

---

## How we built it

- **Development tools:** VS Code, git, Python 3.12 (targets 3.10+).
- **Libraries and frameworks:** **none.** Python standard library only — `json`,
  `math`, `re`, `dataclasses`, `collections`, `pathlib`. No PyTorch, no
  Transformers, no scikit-learn, no vector database. Retrieval is a hand-built
  BM25 inverted index with an IDF-weighted phrase index over the catalog.
- **APIs used:** **none in the submitted agent.** An external LLM API was used
  during development only, to generate the frozen perturbation test sets. It is
  not required to run, score, or reproduce the agent, and no credentials are
  present in the repository.
- **Datasets and assets:** the organizer's frozen 50,000-product
  `Clothing_Shoes_and_Jewelry` catalog and 200 labelled public sessions, from
  Amazon Reviews 2023. The two additional session pools are derived from that
  frozen catalog, which is never modified.

## Disclosure

| | |
|---|---|
| Network access required | **No** |
| Model / API | None |
| Token usage | 0 prompt, 0 completion |
| Estimated cost | **$0.00** |
| Per-turn latency | median **12.8 ms**, p95 97 ms, p99 128 ms, max 168 ms |
| Start-up (one-time index build) | 12.1 s |
| Full 200-session evaluation | 13.4 s |

Absolute timings are hardware-dependent — this run is on a machine roughly 1.5×
slower than an earlier recorded one. Zero tokens, zero cost and no network are
properties of the agent, not of the machine: it reads no environment variable and
opens no socket, so it runs unchanged under whatever CPU, memory, timeout and
network restrictions the organizer imposes.

Runs are deterministic and byte-identical across repeats.

## Challenges

The hardest problem was not retrieval — it was **not fooling ourselves**. The
public set is small, templated, and leaks the target's category verbatim into the
customer's opening. A 50-line agent that exploits only that leak scores 0.831, so
almost any work looks like progress. Separating real improvement from overfitting
meant building held-out product pools and blind perturbation sets *before*
trusting any number, and reverting two changes that looked like wins on the data
we had tuned on.

## Team contributions

| Member | Contribution |
|---|---|
| **Ng Ming Yu** | Retrieval core and ranking — catalog indexing, BM25, the IDF-weighted phrase index, and the commit-to-one-pick recommendation policy (`src/catalog.py`, `src/shopping_agent.py`) |
| **Aeson Ng** | Dual-track intent routing — buying/browsing inference and the per-track retrieval weights (`src/routing.py`) |
| **Seng Boon Kiat** | Dialog state machine — weighted slots, intent-override handling, retraction, and the elimination-validity rule (`src/dialog.py`) |
| **Nathan Quek Xiu Han** | Runtime orchestration and personalized context distillation — the `focus → broaden → diversify` supervisor and the profile layer (`src/strategy.py`, `src/profile.py`) |
| **Nguyen Duy Minh** | Evaluation and robustness — the held-out session pools, the seven blind perturbation sets, the overfitting audit, and the test suite (`heldout_eval.py`, `reports/`, `tests/`) |

## What's next

A locally bundled pretrained sentence encoder (our corpus-trained one failed, but
a pretrained one is a different proposition, and vocabulary mismatch is exactly
what our weakest style measures); a learned `P(purchase | category, profile)`
prior to replace the hand-set popularity weight; and a permanent held-out tuning
split so weights are never fitted on the set they are reported on.
