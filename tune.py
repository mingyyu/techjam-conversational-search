"""Weight sweep. Builds the index once and re-scores in-process.

All ranking weights live in ``TRACKS`` in ``src/routing.py``; ``set_tracks``
patches them in place and restores them after each measurement. The other
high-leverage knob, ``COMMIT_TURNS``, lives in ``src/shopping_agent.py``.
"""
import dataclasses

from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
import src.routing as rt
import src.shopping_agent as sa
from starter.agent import Agent

CATALOG = "data/catalog.jsonl"
samples = load_jsonl("data/public_set.jsonl")
ids, cats, prods = catalog_index(CATALOG)
agent = Agent(CATALOG)

BASE = {name: dataclasses.asdict(track) for name, track in rt.TRACKS.items()}


def set_tracks(**overrides):
    """Override a field on both tracks, or one track via a `buying_`/`browsing_` prefix."""
    updated = {}
    for name, fields in BASE.items():
        fields = dict(fields)
        for key, value in overrides.items():
            if key.startswith(name + "_"):
                fields[key[len(name) + 1:]] = value
            elif not key.startswith(("buying_", "browsing_")):
                fields[key] = value
        updated[name] = rt.Track(**fields)
    rt.TRACKS.clear()
    rt.TRACKS.update(updated)


def score(label, **overrides):
    set_tracks(**overrides)
    result = evaluate(agent, samples, ids, cats, prods)
    print(f"{label:34s} HR={result['hit_rate_at_10']:.3f} MRR={result['mrr']:.4f} "
          f"MTTC={result['mttc']:.3f} score={result['recommended_technical_score']:.5f}",
          flush=True)
    set_tracks()
    return result["recommended_technical_score"]


if __name__ == "__main__":
    score("shipped")
    for weight in (2.0, 4.0, 8.0, 12.0, 20.0):
        score(f"w_popularity={weight}", w_popularity=weight)
    for weight in (0.15, 0.3, 0.6, 1.0):
        score(f"w_bm25={weight}", w_bm25=weight)
    for weight in (3.0, 5.0, 10.0, 14.0):
        score(f"w_phrase={weight}", w_phrase=weight)

    # COMMIT_TURNS is the highest-leverage knob; see reports/commit_depth.json.
    for depth in (0, 1, 2, 3, 4, 5):
        sa.COMMIT_TURNS = depth
        score(f"COMMIT_TURNS={depth}")
    sa.COMMIT_TURNS = 3
