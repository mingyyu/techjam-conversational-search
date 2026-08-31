"""Runtime orchestration.

A fixed pipeline is fine when the opening turn routes correctly. It fails badly
when it does not: the agent keeps mining a pool that cannot contain the target
and spends all ten turns being confidently wrong. Stress testing showed exactly
that, and it was worth up to 0.42 of score.

This module watches for the stall and re-orchestrates the pipeline in response.
Three modes, escalating:

``focus``      the category pool, ranked by constraints. The default.
``broaden``    routing is suspect -- drop the hard category filter and treat the
               category as a soft ranking signal over the whole catalog.
``diversify``  the pool is right but the ordering is not converging -- spread
               picks across distinct sellers so each turn covers more ground.

Escalation is driven by observable state only: turns elapsed, whether new
information arrived, and how much of the pool is already exhausted. The agent
never sees whether it was correct, so the trigger cannot depend on that.
"""

from __future__ import annotations

from dataclasses import dataclass

FOCUS = "focus"
BROADEN = "broaden"
DIVERSIFY = "diversify"

# Turns of no new information before routing is treated as suspect.
STALL_LIMIT = 2

# Fraction of the pool consumed before spreading picks out.
EXHAUSTION_RATIO = 0.5


@dataclass
class Orchestrator:
    mode: str = FOCUS
    stalled_turns: int = 0
    transitions: list[str] = None

    def __post_init__(self) -> None:
        if self.transitions is None:
            self.transitions = []

    def observe(self, turn: int, learned_something: bool, pool_size: int,
                shown_count: int) -> str:
        """Update and return the mode for this turn."""
        if learned_something:
            self.stalled_turns = 0
        elif turn > 1:
            self.stalled_turns += 1

        previous = self.mode

        if pool_size and shown_count >= EXHAUSTION_RATIO * pool_size:
            # The pool is running out before the target appeared. Either it is
            # the wrong pool or the ordering is poor; widening costs nothing at
            # this point because the narrow option is nearly spent.
            self.mode = BROADEN
        elif self.stalled_turns >= STALL_LIMIT and self.mode == FOCUS:
            self.mode = BROADEN
        elif self.mode == BROADEN and self.stalled_turns >= STALL_LIMIT * 2:
            self.mode = DIVERSIFY

        if self.mode != previous:
            self.transitions.append(f"turn {turn}: {previous} -> {self.mode}")
        return self.mode

    def reset(self) -> None:
        self.mode = FOCUS
        self.stalled_turns = 0
        self.transitions = []
