"""Harness shim.

`evaluator/local_evaluator.py` imports `starter.agent`, so this module has to
exist under that name. It re-exports the submitted `Agent` from `agent.py`
rather than defining its own, so the dependency runs bundle -> harness and never
the other way: `agent.py` plus `src/` is self-contained without this file.
"""
from __future__ import annotations

from agent import Agent

__all__ = ["Agent"]
