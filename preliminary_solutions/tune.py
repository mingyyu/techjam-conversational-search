"""Weight sweep. Builds the index once and re-scores in-process."""
import itertools, json
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
import src.shopping_agent as sa
from starter.agent import Agent

samples = load_jsonl('data/public_set.jsonl')
ids, cats, prods = catalog_index('data/catalog.jsonl')
agent = Agent('data/catalog.jsonl')

grid = list(itertools.product(
    [7.0, 12.0, 20.0, 32.0],   # W_PHRASE
    [1.0, 2.0],                # W_BM25
    [0.15, 0.55, 1.0],         # W_POPULARITY
))
best = None
for wp, wb, wpop in grid:
    sa.W_PHRASE, sa.W_BM25, sa.W_POPULARITY = wp, wb, wpop
    r = evaluate(agent, samples, ids, cats, prods)
    score = r['recommended_technical_score']
    print(f"phrase={wp:<5} bm25={wb:<4} pop={wpop:<5} -> HR={r['hit_rate_at_10']:.3f} "
          f"MRR={r['mrr']:.4f} MTTC={r['mttc']:.3f} score={score:.5f}", flush=True)
    if best is None or score > best[0]:
        best = (score, (wp, wb, wpop))
print("BEST", best)
