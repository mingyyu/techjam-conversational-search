# Conversational Shopping Agent

> 🛍️ **50,000 products · 10 turns · zero API calls · zero tokens · zero cost**

Conversational Shopping Agent is a multi-turn shopping system that identifies a
customer's intended product in a frozen 50,000-item Amazon catalog. It runs
entirely on the Python standard library and reaches a **TechnicalScore of
0.977275** on the official evaluator.

---

## Inspiration

*💡 Shopping search should understand a conversation, not just a query.*

Real shoppers rarely express everything they want in one perfectly structured
sentence. They explore, add requirements over time, decline questions, change
their minds, and sometimes replace an earlier preference entirely. We wanted to
build an agent that could follow that evolving intent and still retrieve the
right product quickly.

The challenge also created an unusually interesting engineering constraint: the
agent may receive only a message, a turn number, and an anonymised profile. It is
never told whether the customer is browsing, buying, or about to override an
earlier preference. It must infer the situation from the conversation itself.

At the same time, we did not want to mistake benchmark familiarity for genuine
intelligence. The public set contains only 200 templated sessions and exposes the
target category in the opening message. A 50-line agent that exploits only that
leak scores 0.831, making almost any additional work look like progress. That
inspired a second goal alongside retrieval: build evaluation tools capable of
proving our own ideas wrong before the private set did.

---

## What it does

*🧭 It listens, remembers, clarifies, retrieves, and changes strategy when the
conversation stops making progress.*

### Infers buying versus browsing intent

The agent re-evaluates intent on every turn. It reads requirement markers such
as “must be” and “I need,” exploration markers such as “still exploring” and
“open to,” and—weighted above both—whether a genuinely filterable constraint has
entered the conversation state.

A buying turn leans harder on the customer's own vocabulary. A browsing turn
spreads recommendations across sellers to cover more ground. State deliberately
beats wording: if a session opens with “just browsing” and later states a firm
requirement, the agent treats it as buying regardless of how it began.

### Maintains multi-turn state and handles intent overrides

Requirements accumulate across turns and are weighted by how firmly the shopper
asserted them. When the customer overrides their intent, the agent removes only
revocable preferences and preserves requirements that were restated as firm.

The hardest edge case occurs *before* the override arrives. The protocol
suppresses conversion until that later turn, so an earlier recommendation can
silently contain the target. The agent detects that state and suspends its
elimination memory instead of permanently discarding the answer. Fixing this
single rule moved the Intent Override scenario from 0.233 to approximately
1.000.

### Clarifies proactively by information gain

The agent recommends and asks a question in the same turn. The protocol permits
both, so spending a whole turn on a question alone costs time without adding any
benefit.

It starts with an open question while open questions are still productive, then
selects the attribute that divides the remaining candidate pool closest to
50/50. That maximises expected elimination whichever way the customer answers.
Declined attributes are remembered and never asked again.

### Re-orchestrates retrieval at runtime

A fixed pipeline cannot recognise one of its worst failure modes: the opening
message was routed to the wrong product family, so every later turn searches a
pool that cannot contain the target. A lightweight supervisor watches observable
signals—turns elapsed, whether new information arrived, and how much of the pool
has been exhausted—and escalates through:

> **focus → broaden → diversify**

### Produces strong results without an online model

On the official 200-session evaluator, the agent achieves:

| Metric | Result |
|---|---:|
| **TechnicalScore** | **0.977275** |
| **Hit Rate@10** | **1.000** |
| **MRR** | **0.99125** |
| **MTTC** | **2.005** |
| Rank-1 conversions | **197 / 200** |

That is a **9.2× improvement** over the published 0.1067 baseline, with a
Hit Rate@10 of 1.000 in all four scenario types.

---

## How we built it

*⚙️ A deterministic retrieval stack, built from first principles with the
standard library.*

### Retrieval and ranking

At startup, the agent reads the frozen catalog and builds complementary in-memory
indexes:

- a category index for high-recall product-family filtering;
- an IDF-weighted phrase index for precise catalog-attribute matches;
- an intent-card index containing only constraints the simulated shopper could
  actually disclose; and
- a hand-built BM25 inverted index for robust lexical fallback.

The ranker combines exact phrase evidence, intent-card evidence, BM25 relevance,
category fit, profile-derived preferences, and a measured popularity prior. The
entire process is deterministic and byte-identical across repeated runs.

### Technology

- **Development tools:** VS Code, Git, and Python 3.12, targeting Python 3.10+.
- **Libraries and frameworks:** none. The submitted agent uses only the Python
  standard library, including `json`, `math`, `re`, `typing`, `dataclasses`,
  `collections`, and `pathlib`. There is no PyTorch, Transformers,
  scikit-learn, or vector database.
- **APIs:** none in the submitted agent. A separate external LLM was used only
  during development to generate frozen perturbation test sets. It is not needed
  to run, score, or reproduce the agent, and the repository contains no
  credentials.
- **Datasets and assets:** the organiser's frozen 50,000-product
  `Clothing_Shoes_and_Jewelry` catalog and 200 labelled public sessions from
  Amazon Reviews 2023. Our additional session pools are derived from the same
  catalog, which is never modified.

### Runtime disclosure

| | |
|---|---:|
| Network access required | **No** |
| Model / API | None |
| Token usage | 0 prompt, 0 completion |
| Estimated cost | **$0.00** |
| Per-turn latency | median **12.8 ms**, p95 97 ms, p99 128 ms, max 168 ms |
| Representative one-time index build | 12.1 s |
| Full 200-session evaluation | 13.4 s |

Absolute timings are hardware-dependent; the recorded run used a machine roughly
1.5× slower than an earlier one. Zero tokens, zero cost, and zero network access
are properties of the agent itself: it reads no environment variable and opens
no socket, so it runs unchanged under the organiser's CPU, memory, timeout, and
network restrictions.

### Team contributions

| Member | Contribution |
|---|---|
| **Ng Ming Yu** | Retrieval core and ranking: catalog indexing, BM25, the IDF-weighted phrase index, and the commit-to-one-pick recommendation policy (`src/catalog.py`, `src/shopping_agent.py`) |
| **Aeson Ng** | Dual-track intent routing: buying/browsing inference and per-track retrieval weights (`src/routing.py`) |
| **Seng Boon Kiat** | Dialog state machine: weighted slots, intent-override handling, retraction, and the elimination-validity rule (`src/dialog.py`) |
| **Nathan Quek Xiu Han** | Runtime orchestration and personalised context distillation: the `focus → broaden → diversify` supervisor and profile layer (`src/strategy.py`, `src/profile.py`) |
| **Nguyen Duy Minh** | Evaluation and robustness: held-out session pools, seven blind perturbation sets, the overfitting audit, and the test suite (`heldout_eval.py`, `reports/`, `tests/`) |

---

## Challenges we ran into

*🧩 The hardest problem was not retrieval—it was avoiding false confidence.*

The public set is small, templated, and unusually easy to overfit. With roughly a
dozen adjustable parameters, polishing the visible score could easily produce a
system that failed when the private evaluation swapped in different users and
target products. We therefore built held-out product pools and blind language
perturbations before trusting any improvement, and reverted two changes that
looked like wins on tuned data.

Natural language introduced another class of traps. A template-matching bug could
silently discard an otherwise parseable turn. Semicolons could either separate
two disclosed constraints or belong inside one catalog field. Singular product
names had to resolve plural catalog labels without mangling words such as
“dress.” Each fix had to improve robustness without coupling runtime code to the
evaluator.

Intent overrides were especially subtle because the simulator can withhold
conversion until the override lands. Ordinary elimination logic is correct in a
normal session but destructive in that temporary state. We had to make the agent
know when *not* to trust its own earlier exclusions.

We also prototyped an LLM parser that handled free-form messages better, but it
created the wrong operational trade-off. It required a live endpoint, raised
per-turn latency from approximately 9 ms to approximately 1.3 s, and, when the
endpoint was unavailable, would burn the full timeout for every message—a
projected 13+ hours across 800 sessions. Since the organiser may disable network
access, a solution dependent on that endpoint could score zero despite looking
better locally.

---

## Accomplishments that we're proud of

*🏆 We built not only a strong agent, but also the test suite that could prove it
was wrong.*

### Generalising to unseen products

The public evaluation contains 200 sessions, while the private evaluation uses
800 sessions with different users and different targets. To measure product
generalisation directly, we created two additional 800-session pools from the
frozen catalog:

- a **matched pool** reproducing the purchase-popularity profile of real Amazon
  orders, our closest available proxy for the private set; and
- a **long-tail pool** sampled uniformly as an adversarial test of whether the
  popularity prior was load-bearing.

| Pool using official wording | TechnicalScore | Hit@10 |
|---|---:|---:|
| Public 200—the visible set | 0.977275 | 1.000 |
| **Matched 800—unseen products, purchase-like popularity** | **0.957786** | **0.989** |
| Long-tail 800—uniform adversarial sample | 0.925450 | 0.968 |

**0.9578 is our honest private-set estimate, not the visible public score of
0.9773.** Even if private targets were uniformly long-tail—which the
purchase-record-based specification makes unlikely—the score would decline by
only another 0.032 and Hit@10 would remain 0.968. There is no performance cliff
within that range.

### Surviving language variation

The specification allows natural-language paraphrasing, and a real shopper will
not follow a simulator template. Using a separate model that was deliberately
never shown our parser, we generated seven blind perturbation styles: natural,
terse, rambling, typos, indirect, renamed categories, and information loss.

That harness uncovered issues the clean score could not:

- The inherited version's Hit@10 collapsed to **0.535** under light rewording,
  missing the target in almost half of all sessions despite a clean score of
  0.907.
- A ranking change that improved every style we had tuned on was
  *anti-correlated with target popularity* and would have reduced private-set
  performance, so we reverted it.
- A template-matching bug silently discarded fully parseable turns.

| Style on public 200 | Inherited | Ours |
|---|---:|---:|
| Official wording | 0.9068 | **0.977275** |
| Rambling | 0.8815 | **0.936576** |
| Renamed categories | 0.8832 | **0.931116** |
| Terse | 0.4520 | **0.890260** |
| Typos | 0.4520 | **0.882156** |
| Natural | 0.4967 | **0.841981** |
| Indirect | 0.4819 | **0.802883** |

### Showing that the architecture—not one lucky tuning—is carrying the result

We perturbed every fitted constant simultaneously by a uniform random factor. At
±25%, the worst of eight draws lost only 0.003 and Hit@10 remained 1.000. At
±50%, one draw even scored above the shipped configuration. The parameters sit
on a plateau rather than a fragile peak.

---

## What we learned

*📚 Measurement changed our architecture more than intuition did.*

### Returning more products is not always safer

The evaluator stops a session as soon as the target enters the top ten, fixing
the product's rank at its first appearance. A low-ranked early guess is therefore
the opposite of a hedge: it permanently locks in that low rank.

For one session out of 200, an extra turn costs 0.0001 of score, moving a target
from rank 3 to rank 1 gains 0.0010, and losing a hit costs 0.0025. While the
customer can still disclose useful information, the agent returns only its best
candidate and asks a question. Once disclosures are exhausted, it returns ten
products and sweeps the remaining pool.

That policy moved the public score from 0.9095 to **0.9773** and the matched-800
score from 0.8901 to **0.9578**, with public Hit@10 still at 1.000. Every one of
the sixteen pool-by-wording combinations we measured improved.

The three-turn commitment window comes from the simulator rather than a fitted
constant: an intent card contains at most four constraints and each reply
releases two, so no new information arrives after turn 3. Beyond that point, the
Hit@10 curve turns over exactly where the evaluator's incentives predict.

### Offline reliability can outweigh a higher local NLP score

Our LLM parser scored **0.839** under full rewording, outperforming the offline
parser. We removed it anyway because its network dependency and timeout behavior
made it unsuitable for the grading environment. A solution that cannot run where
it is evaluated scores zero, whatever it scores on a developer's machine.

### Some headroom cannot be reached under the protocol

With Hit@10 at 1.000 and 197 of 200 conversions at rank 1, only 0.019 of score
remains, while MTTC cannot fall below 1.390. An intent-override session is
forbidden from converting before the override arrives on turn 3 or 4. Knowing
that boundary stopped us from spending time optimising against a wall.

### Negative results are still results

We built, measured, and disabled four ideas:

- cross-category browsing pools;
- a corpus-trained LSA semantic encoder;
- a larger personalisation weight; and
- a confidence gate that committed only when the top candidate's score margin
  crossed a threshold—which performed worse at every tested threshold.

Documenting those failures was as important as documenting the wins.

---

## What's next for Conversational Shopping Agent (Andesine)

*🚀 The next version will improve semantic reach without giving up offline,
deterministic execution.*

We plan to explore:

1. **A locally bundled pretrained sentence encoder.** Our corpus-trained encoder
   failed, but a pretrained model is a different proposition, and vocabulary
   mismatch is exactly what the weakest perturbation styles expose.
2. **A learned `P(purchase | category, profile)` prior.** This would replace the
   manually weighted popularity signal with a measured purchase-likelihood
   model.
3. **A permanent held-out tuning split.** We want to guarantee that weights are
   never fitted on the same sessions used to report results.

The goal is not merely a higher visible score. It is an agent that remains fast,
private, reproducible, and useful when real shoppers stop speaking like a test
template.
