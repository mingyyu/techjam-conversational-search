"""Submission entry point.

`docs/submission_rules.md` asks for an `agent.py` at the bundle root exporting
`Agent`, alongside `requirements.txt`, `README.md` and `src/`. This module is
that file, and it is self-contained: it depends only on `src/`, so the four
submitted artifacts run on their own.

The official harness imports `starter.agent` instead. That module re-exports
this class rather than the other way round, so nothing in the submitted bundle
depends on a package the bundle does not contain.

    from agent import Agent          # submitted bundle, and any host
    from starter.agent import Agent  # evaluator/local_evaluator.py

`Agent` is a thin adapter. All behaviour lives in `src.shopping_agent`; keeping
the protocol surface separate from the implementation means the harness contract
is readable in one screen and cannot drift as the ranker changes.
"""
from __future__ import annotations

from src.shopping_agent import ShoppingAgent

__all__ = ["Agent"]


class Agent:
    """The interface `docs/agent_api_contract.json` specifies."""

    def __init__(self, catalog_path: str = "data/catalog.jsonl") -> None:
        self._impl = ShoppingAgent(catalog_path)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._impl.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int,
                top_k: int) -> dict:
        return self._impl.respond(session_id, user_message, turn, top_k)
