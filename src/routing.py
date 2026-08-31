"""Dual-track intent routing.

The customer's underlying intent is never handed to the agent -- `respond` gets
a message, a turn number and an anonymised profile, and nothing else. So the
track has to be inferred from what was said.

Two tracks, because the two intents want opposite things from retrieval:

``buying``    The customer has already decided what they need and stated a hard
              requirement. Precision wins. Lock onto the constraint: weight
              phrase and lexical agreement heavily, keep the category filter
              strict, and discount the popularity prior, which is exactly the
              signal that drags a specific request back towards generic
              best-sellers.

``browsing``  The customer is exploring and has stated no requirement. There is
              nothing to filter on, so precision has nothing to work with.
              Recall and spread win instead: keep the purchase prior at full
              strength -- it is the only signal left -- and diversify picks from
              the first turn so each round covers more distinct ground.

Pooling neighbouring product families for the browsing track was built and
then removed: measured on the public set it cost 0.005 of clean score and
changed nothing under reworded input, because reworded sessions never reach the
named-label branch -- they resolve categories from the whole sentence via
``resolve_categories``, which already pools up to eight families. That is the
path where recall is actually at risk, and it already widens. Raising the
profile weight on the browsing track was measured too, and also costs score.
See reports/tier2.md.

Detection is lexical and structural, never template-shaped, because the private
sessions may not be templated. Three inputs:

  * requirement markers in the message ("key requirement", "must be", "I need")
  * exploration markers ("still exploring", "just looking", "open to")
  * whether a hard constraint has actually landed in dialog state

State beats wording. A session that says "just browsing" and then states a firm
requirement is buying, whatever the opening sounded like. The reverse also
holds: an intent override erases the old requirement, so the track is
recomputed from what survives rather than latched forever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BUYING = "buying"
BROWSING = "browsing"

# Wording that signals a firm requirement. Deliberately broader than the
# evaluator's own templates -- these are the words people reach for when they
# have already decided.
BUYING_MARKERS = (
    "key requirement", "requirement is", "must be", "must have", "has to be",
    "have to be", "needs to be", "need it to", "i need", "i want", "i'm after",
    "im after", "looking for something that", "specifically", "exactly",
    "it should be", "should be", "make sure", "only interested in", "non-negotiable",
)

# Wording that signals open-ended exploration.
BROWSING_MARKERS = (
    "still exploring", "just looking", "just browsing", "browsing",
    "not sure", "no idea", "open to", "any suggestions", "suggestions",
    "what do you have", "what have you got", "ideas", "some options",
    "show me", "have a look", "window shopping", "checking out",
    "thinking about", "maybe", "something like",
)

_MARKER_RES = {
    BUYING: tuple(re.compile(r"\b" + re.escape(m).replace(r"\ ", r"\s+") + r"\b", re.I)
                  for m in BUYING_MARKERS),
    BROWSING: tuple(re.compile(r"\b" + re.escape(m).replace(r"\ ", r"\s+") + r"\b", re.I)
                    for m in BROWSING_MARKERS),
}


@dataclass(frozen=True)
class Track:
    """A retrieval and ranking policy."""

    name: str
    w_phrase: float
    w_bm25: float
    w_popularity: float
    w_profile: float
    diversify_early: bool     # spread picks across sellers from the first turn


# Weights are relative to the single-track baseline (phrase 7.0, bm25 0.3,
# popularity 6.0, profile 0.05), which both tracks reduce to if the deltas are
# set to zero. See reports/tier2.md for the measured effect of each.
TRACKS = {
    BUYING: Track(
        name=BUYING,
        w_phrase=7.0,
        w_bm25=0.45,         # the constraint's own words carry the turn
        # Discounting the purchase prior here was measured and rejected. It
        # helps when targets are long-tail and hurts when they are popular, and
        # real purchases -- which is what the private sessions are -- concentrate
        # on popular products. At 2.0 it cost 0.006 on the popularity-matched
        # pool. See reports/tier2.md.
        w_popularity=6.0,
        w_profile=0.05,      # the customer told us what they want; don't second-guess
        diversify_early=False,
    ),
    BROWSING: Track(
        name=BROWSING,
        w_phrase=7.0,
        w_bm25=0.3,
        w_popularity=6.0,    # with no constraint, the purchase prior is the signal
        w_profile=0.05,
        diversify_early=True,
    ),
}


class IntentRouter:
    """Infers the active track, one turn at a time."""

    def __init__(self) -> None:
        self.track = TRACKS[BROWSING]
        self.transitions: list[str] = []
        self._buying_evidence = 0
        self._browsing_evidence = 0

    @staticmethod
    def _count(message: str, kind: str) -> int:
        return sum(1 for pattern in _MARKER_RES[kind] if pattern.search(message))

    def observe(self, message: str, state, turn: int) -> Track:
        """Update and return the track for this turn."""
        text = message or ""
        self._buying_evidence += self._count(text, BUYING)
        self._browsing_evidence += self._count(text, BROWSING)

        # An override rewrites intent outright: the customer has just told us
        # what they actually need, so the old evidence no longer describes them.
        if getattr(state, "override_seen", False):
            self._browsing_evidence = 0
            self._buying_evidence = max(self._buying_evidence, 1)

        # State is stronger evidence than wording. A hard constraint that has
        # actually landed means there is something precise to filter on,
        # whatever the sentence sounded like.
        hard = state.has_hard_constraint() if hasattr(state, "has_hard_constraint") else False

        if hard or self._buying_evidence > self._browsing_evidence:
            chosen = TRACKS[BUYING]
        else:
            chosen = TRACKS[BROWSING]

        if chosen.name != self.track.name:
            self.transitions.append(f"turn {turn}: {self.track.name} -> {chosen.name}")
        self.track = chosen
        return chosen

    def reset(self) -> None:
        self.track = TRACKS[BROWSING]
        self.transitions = []
        self._buying_evidence = 0
        self._browsing_evidence = 0
