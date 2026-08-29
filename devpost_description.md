# Conversational Shopping Agent

A multi-turn shopping agent that finds a customer's intended product in a frozen
50,000-item Amazon catalog within ten turns — running entirely on the Python
standard library, with **zero API calls, zero tokens, and zero cost**.

**TechnicalScore 0.910201** on the official evaluator (Hit Rate@10 **1.000**,
MRR 0.739, MTTC 1.57), against a published baseline of 0.107.

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

The 200 public sessions are clean templated sentences. The private 800 need not
be. Most of our effort went into finding out how badly that would hurt, rather
than into polishing the number we could see.

We generated **seven perturbation styles** — natural, terse, rambling, typos,
indirect, renamed categories, and information-loss — using a separate model that
was deliberately never shown our parser, so the test could not be tuned against.
Then we built **three session pools**: the public 200, plus 800 unseen products
resampled to match the real purchase-popularity profile, plus 800 sampled
uniformly as an adversarial check on whether our popularity prior was
load-bearing.

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
| official wording | 0.9068 | **0.9102** |
| natural | 0.4967 | **0.8075** |
| terse | 0.4520 | **0.8339** |
| typos | 0.4520 | **0.8380** |
| indirect | 0.4819 | **0.7593** |
| rambling | 0.8815 | **0.8867** |

---

## What we learned

**We cut the LLM on purpose, and measured what it cost.** An LLM parser for
free-form messages scored **0.839** under full rewording — better than our
offline path. We removed it anyway. It needs a live endpoint, and the organizer
reserves the right to score with network access disabled; per-turn latency went
from ~9 ms to ~1.3 s, and with the endpoint unreachable every message burned the
full timeout — a projected 13+ hours for 800 sessions. A solution that cannot run
in the grading environment scores zero, whatever it scores in ours.

**Most of the remaining headroom is not reachable.** Hit@10 is saturated at
1.000, so 87% of what is left sits in MRR. The evaluator stops the turn loop the
instant the target enters the top ten, which means a product's rank is fixed at
first appearance and can never be improved later. MRR is therefore decided by
turn-1 ranking, on a browsing session where the customer has named only a
category — guessing the exact purchased item first out of ~180 peers from one
sentence. Knowing that stopped us burning time on a wall.

**Three ideas we built, measured, and switched off:** cross-category browsing
pools, a corpus-trained LSA semantic encoder, and a raised personalisation
weight. All are documented with their numbers. A negative result measured
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
| Per-turn latency | median **8.8 ms**, p95 61 ms, p99 79 ms |
| Start-up (one-time index build) | 8.2 s |
| Full 200-session evaluation | 6.7 s |

Runs are deterministic and byte-identical across repeats.

## Challenges

The hardest problem was not retrieval — it was **not fooling ourselves**. The
public set is small, templated, and leaks the target's category verbatim into the
customer's opening. A 50-line agent that exploits only that leak scores 0.831, so
almost any work looks like progress. Separating real improvement from overfitting
meant building held-out product pools and blind perturbation sets *before*
trusting any number, and reverting two changes that looked like wins on the data
we had tuned on.

## What's next

A locally bundled pretrained sentence encoder (our corpus-trained one failed, but
a pretrained one is a different proposition, and vocabulary mismatch is exactly
what our weakest style measures); a learned `P(purchase | category, profile)`
prior to replace the hand-set popularity weight; and a permanent held-out tuning
split so weights are never fitted on the set they are reported on.
