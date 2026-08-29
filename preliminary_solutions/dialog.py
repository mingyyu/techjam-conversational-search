"""Conversation state: what the customer has told us, and how much to trust it.

The state machine has to handle three things the brief calls out explicitly:

* **Information accumulation** - each turn may add constraints, which are kept
  with a weight reflecting how strongly they were stated.
* **Intent override** - the customer may revoke an earlier preference. Soft
  preferences are then erased rather than blended, otherwise the revoked
  constraint keeps polluting the ranking.
* **Boundary behaviour** - the customer may decline to answer. That is recorded
  so the agent stops re-asking a dead attribute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Weights by how directly the customer asserted the constraint.
WEIGHT_HARD = 3.0      # stated as a requirement, or asserted after an override
WEIGHT_ANSWER = 2.0    # volunteered in reply to our clarification question
WEIGHT_SOFT = 0.8      # background preference, revocable

OPENING_RE = re.compile(r"looking for\s+(.+)", re.I)
REQUIREMENT_RE = re.compile(r"key requirement is:\s*(.+)", re.I)
DISCLOSURE_RE = re.compile(r"what matters is:\s*(.+)", re.I)
OVERRIDE_RE = re.compile(r"what i need is:\s*(.+)", re.I)
NO_PREFERENCE_RE = re.compile(
    r"don'?t have (?:an? )?(?:additional )?preference for\s+([a-z_]+)", re.I)

OVERRIDE_MARKERS = ("ignore my earlier", "actually,", "instead of")


@dataclass
class Constraint:
    text: str
    weight: float
    revocable: bool = False


@dataclass
class DialogState:
    session_id: str
    profile: dict
    category: str | None = None
    constraints: list[Constraint] = field(default_factory=list)
    shown: set[str] = field(default_factory=set)
    dead_attributes: set[str] = field(default_factory=set)
    override_seen: bool = False
    expects_override: bool = False
    turns: int = 0

    # -- mutation ----------------------------------------------------------

    def add(self, text: str, weight: float, revocable: bool = False) -> bool:
        text = text.strip().rstrip(".")
        if not text:
            return False
        key = text.lower()
        if any(c.text.lower() == key for c in self.constraints):
            return False
        self.constraints.append(Constraint(text, weight, revocable))
        return True

    def apply_override(self, new_value: str) -> None:
        """Erase revocable preferences, then assert the replacement."""
        self.override_seen = True
        self.constraints = [c for c in self.constraints if not c.revocable]
        self.add(new_value, WEIGHT_HARD)

    # -- parsing -----------------------------------------------------------

    def ingest(self, message: str) -> None:
        """Update state from one customer utterance."""
        self.turns += 1
        text = (message or "").strip()
        if not text:
            return

        lowered = text.lower()

        # An override rewrites intent and must be handled before anything else.
        override = OVERRIDE_RE.search(text)
        if override and any(marker in lowered for marker in OVERRIDE_MARKERS):
            self.apply_override(override.group(1))
            return

        # Opening turn: names the product family, and possibly a requirement.
        opening = OPENING_RE.search(text)
        if opening and self.category is None:
            remainder = opening.group(1).strip()

            # "<category>, but I'm still exploring."  -> browsing, no constraint
            explore = re.split(r",\s*but\b", remainder, maxsplit=1)
            head = explore[0]

            # "<category>. <rest>"
            parts = head.split(".", 1)
            self.category = parts[0].strip().rstrip(",")
            tail = parts[1].strip() if len(parts) > 1 else ""
            if not tail and len(explore) == 1:
                # The requirement clause may sit after the category sentence.
                after = text[opening.end():]
                tail = after.strip()

            if tail:
                requirement = REQUIREMENT_RE.search(tail)
                if requirement:
                    self.add(requirement.group(1), WEIGHT_HARD)
                else:
                    # A background preference stated up front. Treat as
                    # revocable: this is exactly what an override cancels.
                    #
                    # An opening that states a preference without framing it as
                    # a requirement, and without signalling open-ended
                    # browsing, is the signature of a session whose intent will
                    # later be rewritten. Recording that changes what we are
                    # allowed to infer from a non-converting turn.
                    self.add(tail, WEIGHT_SOFT, revocable=True)
                    self.expects_override = True
            return

        # Reply to a clarification question.
        disclosure = DISCLOSURE_RE.search(text)
        if disclosure:
            for piece in disclosure.group(1).split(";"):
                self.add(piece, WEIGHT_ANSWER)
            return

        declined = NO_PREFERENCE_RE.search(text)
        if declined:
            self.dead_attributes.add(declined.group(1).lower())
            return

        # Late requirement statement outside the opening turn.
        requirement = REQUIREMENT_RE.search(text)
        if requirement:
            self.add(requirement.group(1), WEIGHT_HARD)

    # -- queries -----------------------------------------------------------

    def active_constraints(self) -> list[Constraint]:
        return self.constraints

    def has_information(self) -> bool:
        return bool(self.constraints)

    def eliminations_are_valid(self) -> bool:
        """Whether a non-converting turn proves the shown products are wrong.

        Normally yes: if the target had been in the list, the session would
        have ended. But while a pending intent override is outstanding the
        protocol suppresses conversion, so a turn can silently contain the
        target. Eliminating those products would discard the answer.
        """
        return self.override_seen or not self.expects_override
