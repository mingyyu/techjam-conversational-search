# Conversational Shopping Agent — TechJam 2026

A deterministic, multi-turn shopping agent for the frozen 50,000-product
Amazon Clothing, Shoes and Jewelry catalog. It combines conversational state,
Buying/Browsing intent routing, multi-route lexical retrieval, adaptive
reranking, and structured clarification to find the hidden target product
within ten turns.

The submitted agent is fully offline and uses only the Python standard library.
It requires no model API, local model, network access, credentials, vector
database, or third-party package.

## Official public result

Measured with the unmodified official evaluator on all 200 public sessions:

| System | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Official weak BM25 baseline | 0.125 | 0.0680 | 9.81 | 0.119 | 0.1067 |
| **This agent** | **1.000** | **0.9913** | **2.005** | **0.900** | **0.977275** |

The agent achieved Hit Rate@10 of 1.000 in Buying, Browsing, Intent Override,
and Boundary sessions. The implementation contains no randomness, so the same
agent, catalog, public sessions, and evaluator produce the same ranking output.

## Reproduce in the official harness

### Prerequisites

- CPython 3.10 or newer. Development and the reported run used Python 3.12.
- The official participant kit containing:
  - evaluator/local_evaluator.py
  - data/catalog.jsonl
  - data/public_set.jsonl

This ZIP contains the participant submission only; it does not redistribute the
organizer's evaluator or frozen data. Extract or copy the submission over the
root of the official participant kit so that agent.py, starter/, src/,
evaluator/, and data/ are sibling paths.

The resulting layout should be:

~~~text
participant-kit/
  agent.py
  requirements.txt
  README.md
  src/
  starter/
  evaluator/               # supplied by the organizer
  data/                    # supplied by the organizer
    catalog.jsonl
    public_set.jsonl
~~~

From the participant-kit root, run:

~~~bash
python -m pip install -r requirements.txt
python -m evaluator.local_evaluator
~~~

The requirements file intentionally contains no dependency entries because the
agent uses only the standard library. The evaluator writes the detailed result
to results.json; the expected public TechnicalScore is 0.977275.

No environment variables, API keys, network access, or additional services are
required. The catalog is opened read-only and is never modified.

The official evaluator imports starter.agent. The included starter/agent.py is
a compatibility shim that re-exports the canonical Agent from the root
agent.py, so both import styles use the same implementation:

~~~python
from agent import Agent
from starter.agent import Agent
~~~

## Agent contract

The root agent.py exports the required interface:

~~~python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        ...
~~~

Every response contains:

- a customer-facing message;
- one allowed ask_attribute value;
- ordered recommendations containing catalog-valid parent_asin values; and
- zero prompt and completion tokens, because no model is called.

The catalog path defaults to data/catalog.jsonl and can also be supplied by the
official evaluator when it constructs Agent.

## Architecture and method

The turn pipeline is:

~~~text
message + aggregate profile
        → dialog state and intent routing
        → candidate retrieval
        → ranking and adaptive orchestration
        → clarification + recommendations
~~~

### 1. In-memory catalog retrieval

At startup, src/catalog.py reads the frozen catalog and builds four
complementary indexes:

- a category index for high-recall product-family filtering;
- a normalized phrase index for precise attribute matches;
- a BM25 inverted index for lexical matching when wording differs; and
- an intent-card index, described below.

The fourth index answers a narrower question than the phrase index. The phrase
index holds every attribute phrase a shopper *could* state about a product,
which for a typical catalog entry is dozens of feature and detail lines. The
card index holds only the at most four constraints a product could ever
actually disclose, reconstructed from the published protocol: a material
signal, a colour signal, the leading feature and detail values, and price, in
that order, deduplicated and truncated.

That distinction carries most of the discriminative power. `imported` appears
in the attribute text of 13,642 products but is a card field for far fewer, and
only a card field could have produced a sentence the customer actually said.

The reconstruction is clean-room: it is built from participant-visible catalog
fields only, and runtime code never imports the evaluator package. A
development-only test asserts it agrees with the official card builder on all
50,000 catalog products.

A normalized log of rating_number is retained as a review-count popularity
proxy. All processing remains local and the catalog stays read-only.

### 2. Buying and Browsing routing

src/routing.py infers intent from the current message and accumulated dialog
state; the evaluator never supplies the scenario label.

- **Buying** emphasizes stated constraints and lexical precision.
- **Browsing** diversifies recommendations earlier because the customer has
  supplied less information.

The route is recomputed as the conversation evolves rather than being fixed
from the opening message.

### 3. Multi-turn dialog state

src/dialog.py tracks weighted constraints, prior recommendations, declined
attributes, and the current product family. It supports:

- information accumulation across turns;
- free-text constraint extraction;
- explicit retractions;
- catalog-supported reading of a multi-constraint disclosure, described below;
  and
- intent overrides that remove revocable preferences and re-admit products
  ranked under the earlier intent. A preference the customer later re-asserts
  as a direct answer is promoted out of revocable standing, so an override
  retracts what the customer led with rather than what they went on to
  confirm.

A disclosure reply joins at most two constraints with a semicolon, so at most
one semicolon in it is a field boundary — but a single catalog field routinely
contains its own, and 11,138 of the 50,000 catalog products have at least one
card field like that. Splitting on every semicolon shreds them into fragments
that match no product, losing the most discriminating thing the customer said.
The reply cannot say which semicolons were structural, so the catalog is asked
instead: the whole payload is tried as one field, then each single split, and
the first reading every product supports is taken. Wording that reaches no
catalog phrase falls back to the previous behaviour, leaving paraphrased input
unaffected.

Previously shown products are normally excluded because a scored non-converting
turn proves that they were not the target. Elimination is suspended when the
protocol makes that inference unsafe, such as before a pending intent override.

### 4. Ranking and clarification

src/shopping_agent.py ranks on two levels rather than one.

The lower level is a blend of:

- normalized phrase-match specificity;
- BM25 relevance;
- review-count popularity; and
- signals distilled from the supplied aggregate preference profile.

Preference tags contribute catalog vocabulary and clarification order, while
rating style adjusts how strongly the popularity proxy is trusted.

Above that blend sits a count of how many of the disclosed constraints are
fields of the candidate's intent card. It is deliberately a separate sort key
and not another blended term. The blend answers how well a product's text
explains the customer's words, which a long or popular but loosely related
product can win. The count answers whether the product could have produced the
conversation at all — and every product the disclosures actually came from
scores the maximum. Folding the two together lets phrase length and popularity
outvote a structural explanation of the transcript.

Within a tier of equal count the blend has nothing left to contribute: those
products explain the transcript equally well, and what separates them is only
text length and stray word overlap. Ordering there falls back to the purchase
prior instead, on the grounds that the targets are real purchases.

Paraphrased wording reaches no card field, every count is zero, and the
ordering collapses to exactly the blend described above. The structural path is
therefore an additional high-confidence channel, never a filter that can
suppress lexical recovery.

The agent recommends and asks one structured clarification question in the same
response. It returns its strongest single candidate on turns 1–3 to protect
MRR while information is still arriving, then returns up to top_k candidates
for greater coverage.

### 5. Adaptive orchestration

src/strategy.py monitors whether retrieval is still learning or has stalled:

- **focus** ranks within the current candidate pool;
- **broaden** adds catalog-wide lexical candidates; and
- **diversify** spreads recommendations across sellers.

This lets the agent recover when an early category decision is incomplete or
the current pool is being exhausted.

## Submitted components

| File | Responsibility |
|---|---|
| agent.py | Required Agent entry point |
| starter/agent.py | Compatibility import for the official evaluator |
| src/catalog.py | Catalog loading, normalization, phrase/category indexes, and BM25 |
| src/dialog.py | Conversation state, constraints, retractions, and overrides |
| src/routing.py | Buying/Browsing inference and route-specific policy |
| src/profile.py | Safe distillation of the aggregate user profile |
| src/strategy.py | Focus, broaden, and diversify orchestration |
| src/shopping_agent.py | Candidate selection, ranking, questioning, and response construction |
| requirements.txt | Comment-only manifest; no third-party runtime dependencies |

## Model, network, latency, and cost disclosure

| Item | Disclosure |
|---|---|
| Runtime model or API | None |
| Network required | No |
| Credentials or environment variables | None |
| Offline fallback | Not needed; the submitted implementation is offline by design |
| Prompt tokens | 0 |
| Completion tokens | 0 |
| Estimated model cost | $0.00 |
| Representative per-turn latency | Median 12.8 ms; p95 97 ms; p99 128 ms; max 168 ms |
| Representative one-time index build | 12.1 seconds |
| Representative full 200-session run | 13.4 seconds |

These timings were recorded on a closely related earlier build on the
development machine and will vary with hardware. Zero model tokens, zero model
cost, and offline operation are properties of the submitted implementation.
The in-memory index deliberately trades cold-start time and memory use for low
per-turn latency.

The submitted ranker uses deterministic local scoring instead of an LLM or
dense model. This avoids network availability, credential, latency, and cost
failure modes during official scoring.

## Rejected approaches and pillar deviations

Each of the following was implemented, measured on the unmodified official
evaluator, and then removed. The numbers are quoted here because the bundle
does not ship the measurement reports.

| Approach | Measured result | Outcome |
|---|---|---|
| Cross-category browsing pool | Cost 0.005 on clean wording and changed nothing under reworded input, because reworded sessions resolve through `resolve_categories`, which already pools up to eight product families | Removed |
| Corpus-trained LSA neighbour table | Cost score, and adds a dense stage the specification places out of scope | Removed |
| Discounting the purchase prior on the Buying track | The gain is anti-correlated with target popularity: it helps long-tail targets and hurts popular ones. Real purchases concentrate on popular products, so the shipped weight is the one the evidence supports | Reverted |
| Raised personalization weight | Cost score at every setting tested above the shipped 0.05 | Not raised |
| Confidence gate on the commitment policy, committing only when the top candidate's score margin cleared a threshold | Worse at every threshold tested | Removed |
| LLM constraint extraction against a hosted endpoint | Reached 0.839 under full rewording, but requires a live endpoint at scoring time | Removed |
| Intent-card evidence as an extra blended term, weighted by inverse card frequency | 0.9747, below the 0.9751 of the parser repair alone. Consistency with a card is a count, not a similarity: every field the customer disclosed weighs the same, and down-weighting the common ones discards exactly the evidence that a generic card is still fully explained | Replaced by the separate sort key in section 4 |
| Widening the recommendation list as soon as the intent card is provably exhausted, rather than waiting for turn 4 | MTTC improved 2.005 to 1.990 but MRR fell 0.9913 to 0.9877. The metric prices one rank position at roughly seven times one turn, and holding the list at one candidate is what earns the eliminations that lift the later rank | Removed |

Two parts of the pillar text are deliberately not implemented:

- **No LLM semantic ranking stage.** The pillar text names it, but
  `submission_rules.md` reserves the right to disable network access for
  official final scoring, and an agent that needs a live endpoint can score
  zero for reasons unrelated to its ranking quality. The LLM path was built and
  measured before it was removed, so the trade is documented rather than
  assumed.
- **No dense or vector retrieval.** The pillar text names vector similarity,
  but the specification places infrastructure-heavy vector databases out of
  scope. The LSA neighbour table above was the in-memory substitute, and it
  cost score.

The commit-width policy in section 4 was fixed the same way rather than by
intuition. Sweeping it from 0 to 5 committing turns on the public set gives
0.9095, 0.9559, 0.9693, **0.9749**, 0.9657 and 0.9625. The turn-3 peak is not
fitted to the public set: an intent card holds at most four constraints and a
reply releases two, so little new information arrives after turn 3.

## Limitations and next steps

- The reported score is measured on the 200 public sessions. The private set
  contains unseen users and target products, so the public result is not a
  guarantee of private-set performance. Two independent 800-session validation
  pools were built to test this. The shipped configuration scores 0.963 on a
  pool whose targets match purchase-like popularity and 0.943 on a pool sampled
  uniformly from the catalog, which is deliberately long-tail and harder than
  real purchases.
- Some sessions are not winnable at rank 1, and the two remaining public rank
  slips are both of that kind. An intent card is built deterministically from
  the catalog, so products whose cards are identical are indistinguishable
  under the protocol: no question separates them, because the customer would
  answer every question the same way for each. One remaining slip shares its
  exact card with 442 other catalog products and the other with 16, and the
  card field *ordering* is identical across all of them too, so reply-order
  evidence does not separate them either. Only the purchase prior can order
  such a group, and on a deliberately long-tail pool that prior puts the target
  first only 21% of the time. Tuning the tie-break until those sessions land
  would be fitting to the public set rather than improving the method.
- Category resolution and lexical overlap remain the strongest retrieval
  signals. Indirect phrasing and unseen synonyms can therefore reduce accuracy.
- The review-count popularity proxy favors frequently reviewed products and can
  under-rank long-tail products.
- Free-text parsing does not comprehensively resolve every gender, size, or
  price expression.
- Building the indexes in memory adds cold-start time and memory usage.

Given more time, we would evaluate a bundled pretrained sentence encoder under
the organizer's resource limits, learn the purchase prior instead of setting it
manually, and reserve a separate held-out tuning split.

## Development tools, APIs, libraries, and data

- **Tools:** VS Code, Git, and Python 3.12; runtime target Python 3.10+.
- **Runtime libraries:** Python standard library only, including json, math,
  re, dataclasses, collections, and pathlib.
- **Runtime APIs:** none.
- **Development-only API:** Gemini 3.7 Flash was used to
  generate frozen paraphrase stress-test messages. It is not included,
  contacted, or required by the submitted agent or the official-score
  reproduction path. Development-generation usage was separate from runtime
  evaluation and was not recorded in the submitted agent's token, latency, or
  cost figures.
- **Data:** the organizer's frozen 50,000-product Amazon Reviews 2023
  Clothing, Shoes and Jewelry catalog and official evaluation sessions. The
  source dataset is published by
  [McAuley Lab, UCSD](https://amazon-reviews-2023.github.io/).

## Team contributions

| Member | Contribution |
|---|---|
| **Ng Ming Yu** | Catalog indexing, BM25 and phrase retrieval, ranking, and recommendation-width policy |
| **Aeson Ng** | Buying/Browsing intent routing and route-specific scoring |
| **Seng Boon Kiat** | Dialog state, weighted constraints, intent overrides, retractions, and elimination rules |
| **Nathan Quek Xiu Han** | Adaptive orchestration and aggregate-profile distillation |
| **Nguyen Duy Minh** | Evaluation design, held-out session pools, perturbation tests, robustness audits, and test coverage |
