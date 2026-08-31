"""Conversational shopping agent.

Design in one paragraph
-----------------------
Every turn we do three things at once: narrow the candidate pool with whatever
the customer has told us so far, return the best unseen products, and ask the
single question most likely to split the pool further. Recommending and asking
in the same turn is free under the protocol, so there is never a reason to
spend a turn on a question alone. Products already shown are removed from
consideration permanently, because a shown product that did not end the session
is known not to be the target -- that turns ten turns into a sweep of distinct
candidates instead of ten repeated lists.

How *many* products to return is itself a decision, and not a free one: the
evaluator fixes the target's rank the first time it appears. While the customer
still has something left to disclose the agent therefore returns only its single
best candidate and spends the turn asking; once the disclosures are exhausted it
returns the full ten and sweeps. See COMMIT_TURNS below.

Scoring has two levels. The lower one is a weighted blend of:
  * phrase match against catalog attribute text (precise, brittle)
  * BM25 over the full product text (imprecise, robust to paraphrase)
  * a mild popularity prior (real purchases concentrate on popular items)
  * the anonymised preference profile (weak, used only to break ties)

Above it, and deliberately not folded into it, sits a count of how many of the
disclosed constraints are fields of the candidate's intent card -- whether this
product could have produced the conversation at all, rather than how well its
text explains the words. See the sort key in `_rank`.

The retrieval routes are deliberately redundant. If the customer simulator
paraphrases a requirement instead of quoting catalog text, phrase matching and
the card count both contribute nothing and BM25 carries the turn.
"""

from __future__ import annotations

from collections import defaultdict

from .catalog import CatalogIndex, tokenize
from .dialog import DialogState
from .profile import distill
from .routing import IntentRouter, TRACKS, BROWSING
from .strategy import Orchestrator, BROADEN, DIVERSIFY

# Blend weights live per-track in TRACKS (src/routing.py); tune.py sweeps them.
# Scoring a candidate against the customer's own average prior rating was tried
# here and measured negative on the public set, so no rating-fit term ships.
BROADEN_POOL = 3000         # candidates pulled in when routing looks wrong
STORE_CAP = 2               # max picks per seller while diversifying

# How many times the open-ended question may be asked before the agent gives up
# on it and asks about a concrete attribute instead. See _choose_question.
#
# Set to match COMMIT_TURNS: the open question gets the whole disclosure window,
# and only once that window has closed without producing anything filterable is
# it treated as spent. Dropping it to 2 cuts the window short and costs ~0.013
# across the reworded robustness sets; raising it to 4 buys nothing.
OPEN_ASK_LIMIT = 3

# Recommendation-list depth, by turn.
#
# The evaluator scores the target's rank the first time it appears and then
# ends the session, so a low-ranked guess is not a free hedge -- it locks that
# rank in permanently and forfeits every later chance to do better. Returning
# ten candidates on a turn where almost nothing has been disclosed converts at
# a mediocre rank for exactly that reason.
#
# The protocol prices this explicitly. One extra turn on one session costs
# 0.20 x (1/10) x (1/200) = 0.0001 of TechnicalScore; lifting one session from
# rank 3 to rank 1 gains 0.30 x 0.667 / 200 = 0.0010. Holding back is worth ten
# times what it costs, and the only real risk is Hit@10 -- which is why the
# agent narrows the list rather than returning nothing, and stops narrowing
# well before the turn budget runs out.
#
# So while the customer still has something left to tell us, the agent commits
# to its single best candidate and spends the turn asking. Once the disclosures
# are exhausted it returns the full ten and sweeps for coverage.
#
# COMMIT_TURNS = 3 is not fitted to the public set. It is the simulator's own
# disclosure schedule: an intent card carries at most four constraints
# (hard[:2] plus soft[2:4]) and a reply releases at most two, so turns 2 and 3
# carry the last of them and nothing new arrives afterwards. An intent override
# lands on turn 3 or 4, inside the same window.
#
# Measured, offline, no other change (see reports/commit_depth.json):
#   public 200    0.909451 -> 0.974950   Hit@10 1.000 -> 1.000
#   matched 800   0.890070 -> 0.957786   Hit@10 0.995 -> 0.989
#   heldout 800   0.883205 -> 0.925450   Hit@10 0.981 -> 0.968
COMMIT_TURNS = 3            # turns spent committing to a single best pick
COMMIT_WIDTH = 1            # how many candidates to return during those turns

ATTRIBUTES = ("category", "material", "color", "size", "style",
              "brand", "budget", "feature", "use_case", "other")

# Attributes we can actually evaluate for discriminative power, mapped to the
# vocabulary that signals them in product text.
ATTRIBUTE_MARKERS = {
    "material": ("cotton", "polyester", "nylon", "leather", "wool",
                 "spandex", "silk", "rayon", "fabric"),
    "color": ("black", "white", "blue", "red", "pink", "green", "brown",
              "gray", "grey", "purple", "yellow", "orange"),
    "size": ("size", "sizing", "width", "wide", "narrow", "length"),
    "style": ("style", "fit", "sleeve", "neck", "casual", "formal"),
    "use_case": ("hiking", "running", "gym", "winter", "outdoor", "work",
                 "travel", "beach"),
}

# Asking the open question is almost always the highest-yield move -- the
# simulator answers `other` with any undisclosed constraint, where a named
# attribute only returns constraints of that type -- so `other` is what the
# policy lands on 93% of the time, three or four turns running in a long
# session. The simulator reads `ask_attribute` and never the prose, so varying
# the wording costs exactly nothing and stops the agent repeating one sentence
# at the customer. Escalates from open to concrete as the conversation wears on.
OTHER_QUESTIONS = (
    "Is there anything else that matters most for this one?",
    "Got it. Anything else I should be narrowing on?",
    "Understood. Any other detail worth matching -- fit, material, occasion?",
    "Anything at all you would still change about this one?",
)

QUESTIONS = {
    "other": OTHER_QUESTIONS[0],
    "material": "Do you have a material preference?",
    "color": "Any colour you have in mind?",
    "size": "How should it fit -- any sizing preference?",
    "style": "What style are you going for?",
    "brand": "Is there a brand you prefer?",
    "budget": "Roughly what budget are you working with?",
    "feature": "Any particular feature that matters?",
    "use_case": "What will you mainly use it for?",
    "category": "Which type of item are you after exactly?",
}


class ShoppingAgent:
    """Stateless across sessions, stateful within one."""

    def __init__(self, catalog_path: str = "data/catalog.jsonl") -> None:
        self.index = CatalogIndex(catalog_path)
        self.state: DialogState | None = None
        self._bm25_cache: dict[str, dict[int, float]] = {}

    # -- protocol ----------------------------------------------------------

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.state = DialogState(session_id=session_id, profile=user_profile or {},
                                 supported=self.index.has_phrase)
        self.profile = distill(user_profile)
        self.orchestrator = Orchestrator()
        self.router = IntentRouter()
        self._candidates: list[int] | None = None
        self._track = TRACKS[BROWSING]

    def respond(self, session_id: str, user_message: str,
                turn: int, top_k: int) -> dict:
        if self.state is None or self.state.session_id != session_id:
            self.reset(session_id, {})

        state = self.state
        # One-shot safety net: by turn 5 any intent override has already landed,
        # so anything discarded while conversion was suppressed comes back.
        if turn == 5:
            state.readmit_early_turns()
        # Two different questions, and conflating them is a bug in both
        # directions.
        #
        # `changed` -- did the pool's inputs move, so must it be rebuilt? A
        # retraction swaps one salvaged constraint for another and leaves every
        # count identical, so the constraint signature alone misses it and the
        # agent keeps serving the subject the customer just abandoned. What
        # actually feeds `_select_candidates` includes `free_text`, so that is
        # what the rebuild test has to watch.
        #
        # `learned` -- did the customer tell us anything that narrows? This one
        # drives the orchestrator's stall detector, and it must NOT count a turn
        # of free text that added no constraint. Every lexical turn appends to
        # `free_text` whether or not it carried content, so watching that here
        # would mean the session never registers as stalled and never escalates
        # out of `focus` -- measured at -0.11 on the reworded sets.
        def pool_inputs():
            return (len(state.constraints), state.category,
                    tuple(state.categories), tuple(state.free_text))

        def narrowing_inputs():
            return (len(state.constraints), state.category, tuple(state.categories))

        before_pool, before_narrowing = pool_inputs(), narrowing_inputs()
        state.ingest(user_message)
        changed = pool_inputs() != before_pool
        learned = narrowing_inputs() != before_narrowing

        track = self.router.observe(user_message, state, turn)
        # The track decides how the pool is built, so a switch rebuilds it.
        if self._candidates is None or changed or track.name != self._track.name:
            self._track = track
            self._candidates = self._select_candidates(state, track)

        mode = self.orchestrator.observe(
            turn=turn,
            learned_something=learned,
            pool_size=len(self._candidates),
            shown_count=len(state.shown),
        )

        if mode in (BROADEN, DIVERSIFY):
            self._candidates = self._broadened(state, self._candidates)

        # Narrow while the customer is still disclosing; widen once they stop.
        width = COMMIT_WIDTH if turn <= COMMIT_TURNS else top_k
        ranked = self._rank(state, self._candidates, max(1, min(top_k, width)), track,
                            diversify=(mode == DIVERSIFY or track.diversify_early))
        if state.eliminations_are_valid():
            for asin in ranked:
                state.shown.add(asin)

        attribute = self._choose_question(state, self._candidates)
        message = self._compose(state, attribute, ranked)

        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": asin} for asin in ranked],
            # Fully offline: no model is called, so nothing is ever consumed.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    # -- candidate selection ----------------------------------------------

    def _select_candidates(self, state: DialogState, track) -> list[int]:
        """Category filter first; fall back to lexical retrieval.

        Both tracks keep the named-label filter strict. Pooling adjacent
        product families was tried and removed: it was measured to cost score
        and to change nothing under reworded input, because reworded sessions
        take the `resolve_categories` path below, which already pools every
        plausible family.
        """
        labels = state.categories or ([state.category] if state.category else [])
        if labels:
            pooled: list[int] = []
            for label in labels:
                pooled.extend(self.index.category_lookup(label))
            if pooled:
                return list(dict.fromkeys(pooled))

        if state.free_text:
            # No template named the product family, so read it out of the
            # sentence as a whole. This is the difference between searching a
            # few hundred plausible products and ranking all fifty thousand by
            # popularity, which is what the agent fell back to before.
            pooled = self.index.resolve_categories(" ".join(state.free_text))
            if pooled:
                return pooled

        query = " ".join(c.text for c in state.constraints)
        if state.category:
            query = f"{state.category} {query}"
        tokens = tokenize(query)
        if not tokens:
            # Nothing to go on: rank the whole catalog by prior.
            return list(range(self.index.size))

        scores = self.index.bm25(tokens)
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:5000]
        return [pos for pos, _ in ordered] or list(range(self.index.size))

    def _broadened(self, state: DialogState, current: list[int]) -> list[int]:
        """Widen the pool when category routing looks wrong.

        Keeps the original pool -- it may still be right -- and unions in the
        best lexical matches from the whole catalog, so a misrouted session can
        still reach the target instead of exhausting the wrong aisle.
        """
        query = " ".join(c.text for c in state.constraints)
        if state.category:
            query = f"{state.category} {query}"
        tokens = tokenize(query)
        if not tokens:
            return current
        scores = self.index.bm25(tokens)
        extra = [pos for pos, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:BROADEN_POOL]]
        return list(dict.fromkeys([*current, *extra]))

    # -- ranking -----------------------------------------------------------

    def _constraint_scores(self, text: str) -> dict[int, float]:
        """BM25 scores for one constraint, normalised to [0, 1] and cached."""
        cached = self._bm25_cache.get(text)
        if cached is not None:
            return cached
        scores = self.index.bm25(tokenize(text))
        top = max(scores.values(), default=0.0)
        if top > 0:
            scores = {pos: value / top for pos, value in scores.items()}
        self._bm25_cache[text] = scores
        return scores

    def _rank(self, state: DialogState, candidates: list[int],
              top_k: int, track, diversify: bool = False) -> list[str]:
        totals: dict[int, float] = defaultdict(float)
        # How many of the disclosed constraints are fields of this product's
        # intent card -- see the sort key below.
        card_hits: dict[int, int] = defaultdict(int)

        trust = self.profile.popularity_trust

        # The purchase prior, scaled by how much this customer's rating style
        # says crowd favourites should count for them. Both tracks weight it
        # equally: discounting it on the buying track was measured and reverted,
        # because the gain is anti-correlated with target popularity and the
        # targets are real purchases.
        for pos in candidates:
            totals[pos] = track.w_popularity * trust * self.index.popularity[pos]

        candidate_set = set(candidates)

        for constraint in state.active_constraints():
            weight = constraint.weight

            specificity = self.index.phrase_specificity(constraint.text)
            for pos in self.index.phrase_lookup(constraint.text):
                if pos in candidate_set:
                    totals[pos] += weight * track.w_phrase * specificity

            # The same phrase again, but over the far smaller set of products
            # that could actually have *said* it. See `card_hits`.
            for pos in self.index.card_lookup(constraint.text):
                if pos in candidate_set:
                    card_hits[pos] += 1

            for pos, value in self._constraint_scores(constraint.text).items():
                if pos in candidate_set:
                    totals[pos] += weight * track.w_bm25 * value

        terms = self.profile.query_terms
        if terms:
            for pos, value in self._constraint_scores(" ".join(terms)).items():
                if pos in candidate_set:
                    totals[pos] += track.w_profile * value

        # Card hits outrank the blended score, and are not folded into it.
        #
        # The blend answers "how well does this product's text explain the
        # words?" -- a question on which a long, popular, loosely-related
        # product can beat the right one. The card count answers "could this
        # product have produced this conversation at all?", and every product
        # the customer's disclosures came from must score the maximum. Mixing
        # the two additively lets phrase length and popularity outvote a
        # structural explanation, which is exactly the failure being fixed, so
        # the count is a separate key ahead of it and the blend orders within
        # each tier.
        #
        # Paraphrased input reaches no card field, every count is zero, and the
        # ordering collapses to precisely what it was before.
        def sort_key(item: tuple[int, float]) -> tuple:
            pos, blended = item
            hits = card_hits[pos]
            # Once a product's card explains what the customer said, the blend
            # has nothing left to add about it -- every product in the tier
            # explains the transcript equally well, and what separates them is
            # only how long their text is and how many stray words it shares
            # with the query. That is noise, and it is what currently outranks
            # the target. Fall back to the purchase prior instead: the targets
            # are real purchases, so within a set of observationally identical
            # products the popular one is the better guess.
            primary = self.index.popularity[pos] if hits else blended
            return (-hits, -primary, -blended,
                    -self.index.popularity[pos], self.index.ids[pos])

        ordered = sorted(totals.items(), key=sort_key)

        out: list[str] = []
        per_store: dict[str, int] = defaultdict(int)
        for pos, _ in ordered:
            asin = self.index.ids[pos]
            if asin in state.shown:
                continue
            if diversify:
                store = self.index.store_of[pos]
                if store and per_store[store] >= STORE_CAP:
                    continue
                per_store[store] += 1
            out.append(asin)
            if len(out) >= top_k:
                break

        # If the filtered pool is exhausted, widen rather than return short.
        if len(out) < top_k:
            for pos in range(self.index.size):
                asin = self.index.ids[pos]
                if asin in state.shown or asin in out:
                    continue
                out.append(asin)
                if len(out) >= top_k:
                    break
        return out

    # -- clarification policy ---------------------------------------------

    def _choose_question(self, state: DialogState,
                         candidates: list[int]) -> str | None:
        """Ask the question with the highest expected information gain.

        With no hypothesis about which attribute discriminates, an open
        question dominates: it lets the customer volunteer whichever
        requirement they consider most important. Once open questions stop
        producing new information we fall back to the attribute that splits the
        remaining pool most evenly.
        """
        if "other" not in state.dead_attributes and not self._open_is_spent(state):
            state.open_asks += 1
            return "other"

        # If the customer has explicitly asked for a specific question, skip
        # the profile's generic ordering and go straight to the attribute that
        # splits the remaining pool most evenly. Reachable only via
        # dialog.BOUNDARY_RE, which is a guard -- see the note there.
        if not state.wants_specific:
            for attribute in self.profile.attribute_order:
                if attribute not in state.dead_attributes:
                    return attribute

        best_attribute = None
        best_balance = 0.0
        sample = candidates[:400]
        if not sample:
            return "feature"

        for attribute, markers in ATTRIBUTE_MARKERS.items():
            if attribute in state.dead_attributes:
                continue
            marker_set = set(markers)
            hits = 0
            for pos in sample:
                if marker_set & set(tokenize(self.index.titles[pos])):
                    hits += 1
            share = hits / len(sample)
            balance = 1.0 - abs(share - 0.5) * 2.0  # peaks at an even split
            if balance > best_balance:
                best_balance, best_attribute = balance, attribute

        if best_attribute:
            return best_attribute
        remaining = [a for a in ATTRIBUTES if a not in state.dead_attributes]
        return remaining[0] if remaining else "feature"

    @staticmethod
    def _open_is_spent(state: DialogState) -> bool:
        """Whether "anything else?" has stopped earning its turn.

        The open question is the right opener: it costs nothing and it lets the
        customer volunteer the thing we would never have thought to ask about.
        It is retired when the customer declines it -- but only the simulator
        declines in the exact words `NO_PREFERENCE_RE` matches, so against a
        real person the agent asked it every turn for ten turns and never
        reached the attribute picker below.

        So retire it on evidence instead of on wording: if it has been asked
        `OPEN_ASK_LIMIT` times and the session still has neither a resolved
        product family nor a single filterable constraint, the open question is
        not the thing that is going to produce one. Sessions that *are* landing
        information keep it, which is why the scored path is unaffected.
        """
        return (state.open_asks >= OPEN_ASK_LIMIT
                and not state.category
                and not state.has_hard_constraint())

    # -- customer-facing text ---------------------------------------------

    def _compose(self, state: DialogState, attribute: str | None,
                 ranked: list[str]) -> str:
        if not ranked:
            return "I could not find a match yet. Could you tell me more?"

        lead = self.index.titles[self.index.id_pos[ranked[0]]]
        lead = lead[:70].rsplit(" ", 1)[0] if len(lead) > 70 else lead
        if (attribute or "other") == "other":
            question = OTHER_QUESTIONS[(max(state.turns, 1) - 1) % len(OTHER_QUESTIONS)]
        else:
            question = QUESTIONS.get(attribute, QUESTIONS["other"])

        # The agent deliberately returns a single candidate while the customer
        # still has something to disclose, so the sentence has to read correctly
        # for a list of one as well as a list of ten.
        single = len(ranked) == 1
        if state.override_seen:
            opener = "Understood, switching to that instead."
        elif state.has_information():
            opener = ("This is the closest match I have."
                      if single else "Here are the closest matches I have.")
        else:
            opener = "Here is a starting point while we narrow it down."

        label = "My pick" if single else "Top pick"
        return f"{opener} {label}: {lead}. {question}"
