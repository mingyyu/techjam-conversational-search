"""Submission entry point.

`docs/submission_rules.md` recommends an `agent.py` at the package root, while
the official harness imports `starter.agent`. Both resolve to the same class;
this module simply re-exports it so either path works.

    from agent import Agent          # recommended submission layout
    from starter.agent import Agent  # what evaluator/local_evaluator.py uses
"""
from __future__ import annotations

from starter.agent import Agent

__all__ = ["Agent"]
