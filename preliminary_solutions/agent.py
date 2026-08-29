"""Submission entry point. Exports `Agent` as required by the harness."""
from __future__ import annotations

from src.shopping_agent import ShoppingAgent


class Agent:
    def __init__(self, catalog_path: str = "data/catalog.jsonl") -> None:
        self._impl = ShoppingAgent(catalog_path)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._impl.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self._impl.respond(session_id, user_message, turn, top_k)
