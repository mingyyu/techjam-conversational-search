"""Conversational shopping agent.

Design in one paragraph
-----------------------
Every turn we do three things at once: narrow the candidate pool with whatever
the customer has told us so far, return the ten best unseen products, and ask
the single question most likely to split the pool further. Recommending and
asking in the same turn is free under the protocol, so there is never a reason
to spend a turn on a question alone. Products already shown are removed from
consideration permanently, because a shown product that did not end the session
is known not to be the target -- that turns ten turns into a sweep of up to a
hundred distinct candidates instead of ten repeated lists.

Scoring is a weighted blend of:
  * phrase match against catalog attribute text (precise, brittle)
  * BM25 over the full product text (imprecise, robust to paraphrase)
  * a mild popularity prior (real purchases concentrate on popular items)
  * the anonymised preference profile (weak, used only to break ties)

The two retrieval routes are deliberately redundant. If the customer simulator
paraphrases a requirement instead of quoting catalog text, phrase matching
contributes nothing and BM25 carries the turn.
"""

from __future__ import annotations

from collections import defaultdict

from .catalog import CatalogIndex, tokenize
from .dialog import DialogState
from .profile import distill
from .routing import IntentRouter, TRACKS, BROWSING
from .strategy import Orchestrator, BROADEN, DIVERSIFY

# Blend weights. Tuned on the 200 public sessions; see tune.py.
W_PHRASE = 7.0      # exact attribute-phrase agreement
W_BM25 = 0.3        # lexical similarity
W_POPULARITY = 6.0  # purchase prior
W_PROFILE = 0.05    # anonymised preference tags
W_RATING_FIT = 0.0  # measured negative on the public set; see README
USE_PROFILE = True          # personalized context distillation
USE_ORCHESTRATION = True    # runtime strategy switching
USE_DUAL_TRACK = True       # buying/browsing intent routing; see src/routing.py
BROADEN_POOL = 3000         # candidates pulled in when routing looks wrong
POOL_TRUST_LIMIT = 3000     # above this the pool was never really narrowed,
                            # so the buying track stops discounting the prior
CROSS_CATEGORY_FLOOR = 400  # browse wider only when the named aisle is this thin
                            # (inert: browsing track ships cross_category=False)
STORE_CAP = 2               # max picks per seller while diversifying
SEMANTIC_EXPANSION = False  # corpus-learned synonym bridging; see reports/robustness.md
EXPANSION_DECAY = 0.45      # a synonym is worth this fraction of a literal hit

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

QUESTIONS = {
    "other": "Is there anything else that matters most for this one?",
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
        self.state = DialogState(session_id=session_id, profile=user_profile or {})
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
        before = (len(state.constraints), state.category, tuple(state.categories))
        state.ingest(user_message)
        changed = (len(state.constraints), state.category, tuple(state.categories)) != before

        track = (self.router.observe(user_message, state, turn)
                 if USE_DUAL_TRACK else TRACKS[BROWSING])
        # The track decides how the pool is built, so a switch rebuilds it.
        if self._candidates is None or changed or track.name != self._track.name:
            self._track = track
            self._candidates = self._select_candidates(state, track)

        mode = self.orchestrator.observe(
            turn=turn,
            learned_something=changed,
            pool_size=len(self._candidates),
            shown_count=len(state.shown),
            has_constraints=state.has_information(),
        ) if USE_ORCHESTRATION else "focus"

        if mode in (BROADEN, DIVERSIFY):
            self._candidates = self._broadened(state, self._candidates)

        ranked = self._rank(state, self._candidates, top_k, track,
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

        Both tracks keep the named-label filter strict. Adjacent-family pooling
        is wired up for the browsing track but ships disabled: it was measured
        to cost score and to change nothing under reworded input, because
        reworded sessions take the `resolve_categories` path below, which
        already pools every plausible family. See src/routing.py.
        """
        labels = state.categories or ([state.category] if state.category else [])
        if labels:
            pooled: list[int] = []
            for label in labels:
                pooled.extend(self.index.category_lookup(label))
            # Cross-category scenario matching, browsing track only, and only
            # when the named aisle is small enough that recall is genuinely at
            # risk. Widening a large pool measurably hurts: it adds candidates
            # that can outrank the target without adding the target.
            if track.cross_category and len(set(pooled)) < CROSS_CATEGORY_FLOOR:
                for label in labels:
                    pooled.extend(self.index.category_neighbours(label))
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
        tokens = tokenize(text)
        if SEMANTIC_EXPANSION:
            scores = self.index.bm25_weighted(
                self.index.expand(tokens, decay=EXPANSION_DECAY))
        else:
            scores = self.index.bm25(tokens)
        top = max(scores.values(), default=0.0)
        if top > 0:
            scores = {pos: value / top for pos, value in scores.items()}
        self._bm25_cache[text] = scores
        return scores

    def _rank(self, state: DialogState, candidates: list[int],
              top_k: int, track, diversify: bool = False) -> list[str]:
        totals: dict[int, float] = defaultdict(float)

        trust = self.profile.popularity_trust if USE_PROFILE else 1.0
        target_rating = self.profile.rating_target if USE_PROFILE else None

        # Slot decay on the popularity prior.
        #
        # The buying track discounts the prior so a stated requirement is not
        # dragged back towards best-sellers. That is only safe when the pool is
        # small enough to mean the requirement actually narrowed something. When
        # the customer's wording left the category vague the pool stays huge,
        # the constraint is not trustworthy, and the prior is the best signal
        # available -- so the discount decays back to the browsing weight in
        # proportion to how little the pool was narrowed.
        w_popularity = track.w_popularity
        if len(candidates) > POOL_TRUST_LIMIT:
            w_popularity = TRACKS[BROWSING].w_popularity

        for pos in candidates:
            score = w_popularity * trust * self.index.popularity[pos]
            if target_rating is not None:
                gap = abs(self.index.avg_rating[pos] - target_rating)
                score += W_RATING_FIT * max(0.0, 1.0 - gap / 2.0)
            totals[pos] = score

        candidate_set = set(candidates)

        for constraint in state.active_constraints():
            weight = constraint.weight

            specificity = self.index.phrase_specificity(constraint.text)
            for pos in self.index.phrase_lookup(constraint.text):
                if pos in candidate_set:
                    totals[pos] += weight * track.w_phrase * specificity

            for pos, value in self._constraint_scores(constraint.text).items():
                if pos in candidate_set:
                    totals[pos] += weight * track.w_bm25 * value

        terms = self.profile.query_terms if USE_PROFILE else [
            str(t) for t in (state.profile.get("preference_tags") or [])]
        if terms:
            for pos, value in self._constraint_scores(" ".join(terms)).items():
                if pos in candidate_set:
                    totals[pos] += track.w_profile * value

        ordered = sorted(
            totals.items(),
            key=lambda kv: (-kv[1], -self.index.popularity[kv[0]], self.index.ids[kv[0]]),
        )

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
        if "other" not in state.dead_attributes:
            return "other"

        # If the customer has explicitly asked for a specific question, skip
        # the profile's generic ordering and go straight to the attribute that
        # splits the remaining pool most evenly. Reachable only via
        # dialog.BOUNDARY_RE, which is a guard -- see the note there.
        if not state.wants_specific and USE_PROFILE:
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

    # -- customer-facing text ---------------------------------------------

    def _compose(self, state: DialogState, attribute: str | None,
                 ranked: list[str]) -> str:
        if not ranked:
            return "I could not find a match yet. Could you tell me more?"

        lead = self.index.titles[self.index.id_pos[ranked[0]]]
        lead = lead[:70].rsplit(" ", 1)[0] if len(lead) > 70 else lead
        question = QUESTIONS.get(attribute or "other", QUESTIONS["other"])

        if state.override_seen:
            opener = "Understood, switching to that instead."
        elif state.has_information():
            opener = "Here are the closest matches I have."
        else:
            opener = "Here is a starting point while we narrow it down."

        return f"{opener} Top pick: {lead}. {question}"
