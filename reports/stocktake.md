> **Historical snapshot.** Taken before dual-track routing and the
> template-gating repair. Kept for the headroom analysis and the
> session-pool characterisation, both of which still hold. For current
> numbers see `../README.md` and `tier2.md`.

# State of the solution — 2026-08-29

All numbers offline. `USE_LLM_EXTRACTION = False`, zero network calls, zero tokens.
29/29 unit tests pass. Clean public score unchanged at 0.906751.

## Datasets

| Set | n | median reviews | median popularity rank in category | what it tests |
|---|---|---|---|---|
| `data/public_set.jsonl` | 200 | 6,846 | 2 | the given dev set |
| `data/matched_sessions.jsonl` | 800 | 1,380 | 7 | **new products, purchase-like popularity** |
| `data/heldout_sessions.jsonl` (Gemini) | 800 | 12 | 82 | popularity-prior stress (uniform catalog sample) |

The Gemini pool samples the catalog uniformly, so its targets are long-tail. Real
sessions are actual Amazon purchases and concentrate on popular products, as the
public set shows. It is a good adversarial set but not a private-set proxy.

## Results

### Wording robustness (public 200, ~92% of messages rewritten)

| Style | Score | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| clean (templated) | 0.9068 | 1.000 | 0.726 | 1.55 |
| categories | 0.8830 | 0.990 | 0.723 | 2.44 |
| typos | 0.8384 | 0.970 | 0.598 | 2.31 |
| terse | 0.8329 | 0.965 | 0.582 | 2.21 |
| natural | 0.8035 | 0.960 | 0.574 | 3.44 |
| rambling | 0.7628 | 0.915 | 0.541 | 3.85 |
| indirect | 0.7519 | 0.895 | 0.531 | 3.75 |
| old qwen set (fitted) | 0.7442 | 0.895 | 0.478 | 3.33 |
| lossy (info removed — report separately) | 0.7876 | 0.935 | 0.660 | 4.90 |

### Product generalisation (templated wording, unseen products)

| Set | Score | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| public 200 | 0.9068 | 1.000 | 0.726 | 1.55 |
| matched 800 | 0.8848 | 0.990 | 0.686 | 1.79 |
| Gemini 800 (long-tail) | 0.8735 | 0.973 | 0.718 | 2.40 |

Per-scenario Hit@10, Gemini long-tail set: boundary 0.750, intent_override 0.958,
browsing 0.984, buying 0.994.

## Reading

- Tier 1 generalises. Held-out rewrites score at or above the set Tier 1 was tuned
  on (0.7519–0.8830 vs 0.7442), so the earlier circularity worry does not bite.
- The Gemini rewrites are milder than the old qwen ones — they often keep template
  phrasing and catalog vocabulary verbatim. Treat 0.744 as the conservative floor.
- Hit@10 is close to saturated everywhere (0.90–1.00). **MRR is where the score is
  lost**: 0.726 clean vs 0.53–0.60 reworded. MRR is 30% of TechnicalScore.
- Boundary is the weakest scenario on unseen long-tail products (0.750), and the
  public set has only 10 boundary sessions, so this was previously invisible.
