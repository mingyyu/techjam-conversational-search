"""End-to-end walkthrough of a single shopping session.

Replays one session against the *unmodified* customer simulator from
`evaluator/local_evaluator.py`, printing what the agent understood at every
turn. Nothing here is staged: the customer's sentences, the hidden target and
the scoring all come from the official evaluator.

The point is to make the internals visible. The agent is a headless backend --
UI work is explicitly out of scope for this challenge -- so the interesting
behaviour lives in the state, not in a chat window:

  * which intent track was inferred, and when it switched
  * slots accumulating turn by turn, with their weights
  * an intent override erasing revocable slots and re-admitting eliminations
  * the clarification question chosen, and which attributes are exhausted
  * the candidate pool the recommendations were drawn from
  * the rank the target landed at, which fixes MRR for the whole session

    python demo.py                          # one session of each scenario type
    python demo.py --scenario intent_override
    python demo.py --sample public_0042
    python demo.py --metrics                # full 200-session evaluator run
"""
from __future__ import annotations

import argparse
import sys

from evaluator.local_evaluator import (
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply, evaluate,
    initial_message, load_jsonl, materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent

RULE = "=" * 78
THIN = "-" * 78


def safe(text) -> str:
    """Catalog titles carry curly quotes and other non-ASCII; a Windows console
    at cp1252 cannot print them. The demo must never die on a product name."""
    return str(text).encode("ascii", "replace").decode("ascii")


def fmt_slots(state) -> str:
    """Slots with their weights, so accumulation and erasure are both visible."""
    if not state.constraints:
        return "(none)"
    rows = []
    for c in state.constraints:
        tag = "revocable" if c.revocable else ("salvaged" if c.salvaged else "firm")
        rows.append(f"{c.text[:44]!r} w={c.weight} [{tag}]")
    return "\n              ".join(rows)


def run_session(agent, sample, ids, cats, products, verbose=True):
    impl = agent._impl
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    eff = {**sample, "intent_card": card, "behavior": behavior}
    disclosed = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"

    session_id = "demo_" + str(sample["sample_id"])
    agent.reset(session_id, sample["user_profile"])
    message = initial_message(eff, coarse_category(cats.get(target, [])), disclosed)

    if verbose:
        print(RULE)
        print("SESSION " + str(sample["sample_id"])
              + "   scenario=" + str(sample["scenario_type"])
              + "   difficulty=" + str(sample.get("difficulty_bucket")))
        print("HIDDEN TARGET  " + target + "  "
              + safe(products[target].get("title", ""))[:52])
        print("               category: " + coarse_category(cats.get(target, [])))
        print("               the agent is never told any of this")
        print("PROFILE        " + safe(sample["user_profile"].get("summary", ""))[:66])
        print(RULE)

    hit_turn = None
    best_rank = None
    for turn in range(1, MAX_TURNS + 1):
        before = len(impl.state.constraints) if impl.state else 0
        eliminated_before = len(impl.state.shown) if impl.state else 0
        response = agent.respond(session_id, message, turn, TOP_K)
        st = impl.state
        track = impl._track
        ranked = normalize_recommendations(response.get("recommendations"), ids)

        if verbose:
            print("")
            print("TURN " + str(turn))
            print("  CUSTOMER    " + safe(message))
            print("  " + THIN[:60])
            extra = ", diversifying" if track.diversify_early else ""
            print("  track       " + track.name.upper()
                  + "  (popularity w=" + str(track.w_popularity)
                  + ", bm25 w=" + str(track.w_bm25) + extra + ")")
            print("  category    " + (st.category or "(unresolved)"))
            print("  slots       " + safe(fmt_slots(st)))
            if len(st.constraints) > before:
                print("              ^ " + str(len(st.constraints) - before)
                      + " new this turn")
            if st.override_seen and eliminated_before:
                print("  override    APPLIED - revocable slots erased, and the "
                      + str(eliminated_before) + " products")
                print("              already ruled out were re-admitted")
            elif st.override_seen:
                print("  override    APPLIED - revocable slots erased")
            # Whether a non-converting turn is allowed to rule products out.
            # While an override is pending the protocol suppresses conversion,
            # so a turn can silently contain the target -- eliminating then
            # would throw the answer away.
            elim = ("ACTIVE" if st.eliminations_are_valid()
                    else "SUSPENDED (override pending)")
            print("  pool        " + str(len(impl._candidates)) + " candidates"
                  + "  | mode=" + impl.orchestrator.mode)
            print("  eliminating " + elim + "  | ruled out so far="
                  + str(len(st.shown)))
            asked = repr(response.get("ask_attribute"))
            spent = (", ".join(sorted(st.dead_attributes))
                     if st.dead_attributes else "none yet")
            print("  asks        " + asked + "   (exhausted: " + spent + ")")
            print("  AGENT       " + safe(response.get("message", ""))[:70])
            for i, asin in enumerate(ranked[:5], 1):
                mark = "   <== TARGET" if asin == target else ""
                title = safe(products.get(asin, {}).get("title", ""))[:40]
                print("     " + str(i) + ". " + asin + "  " + title + mark)

        if override_applied and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            if verbose:
                print("")
                print("  >>> CONVERTED on turn " + str(turn) + " at rank "
                      + str(best_rank) + "  -> reciprocal rank "
                      + format(1.0 / best_rank, ".3f"))
                print("      the evaluator stops the session here, so this rank "
                      "is final")
            break
        if turn == MAX_TURNS:
            break

        ov = eff.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(ov.get("turn", 3)):
            override_applied = True
            if str(ov.get("new_value", "")):
                disclosed.add(str(ov["new_value"]))
            message = str(ov.get("message",
                                 "Actually, please ignore my earlier preference."))
            if verbose:
                print("")
                print("  ** the customer is about to change their mind **")
        else:
            message, boundary_used = customer_reply(
                eff, response.get("ask_attribute"), disclosed, boundary_used)

    if verbose:
        print("")
        print(THIN)
        print("RESULT  hit=" + str(hit_turn is not None)
              + "  turns=" + str(hit_turn or MAX_TURNS)
              + "  rank=" + str(best_rank)
              + "  tokens=0 (fully offline, no network)")
        if impl.router.transitions:
            print("TRACK SWITCHES    " + "; ".join(impl.router.transitions))
        if impl.orchestrator.transitions:
            print("RE-ORCHESTRATION  " + "; ".join(impl.orchestrator.transitions))
        print(RULE)
    return {"hit": hit_turn is not None, "rank": best_rank, "turns": hit_turn}


def show_metrics(agent, samples, ids, cats, products, dataset):
    r = evaluate(agent, samples, ids, cats, products)
    print(RULE)
    print("OFFICIAL EVALUATOR  " + dataset + "  ("
          + str(r["sample_count"]) + " sessions)")
    print(RULE)
    print("  Hit Rate@10   " + format(r["hit_rate_at_10"], ".4f"))
    print("  MRR           " + format(r["mrr"], ".4f"))
    print("  MTTC          " + format(r["mttc"], ".3f"))
    print("  Efficiency    " + format(r["efficiency"], ".4f"))
    print("  TECHNICAL     " + format(r["recommended_technical_score"], ".6f"))
    print("  token usage   " + str(r["reported_token_usage"]))
    print(THIN)
    for name, v in r["scenario_metrics"].items():
        print("  " + name.ljust(16) + " n=" + str(v["sample_count"]).ljust(4)
              + " hit=" + format(v["hit_rate_at_10"], ".3f")
              + "  mrr=" + format(v["mrr"], ".3f")
              + "  mttc=" + format(v["mttc"], ".2f"))
    print(RULE)


def main():
    ap = argparse.ArgumentParser(description="Session walkthrough for the demo video")
    ap.add_argument("--sample", default=None, help="a single sample_id")
    ap.add_argument("--scenario", default=None,
                    choices=["buying", "browsing", "intent_override", "boundary"])
    ap.add_argument("--metrics", action="store_true",
                    help="run the full official evaluator instead")
    ap.add_argument("--dataset", default="data/public_set.jsonl")
    args = ap.parse_args()

    samples = load_jsonl(args.dataset)
    ids, cats, products = catalog_index("data/catalog.jsonl")
    agent = Agent("data/catalog.jsonl")

    if args.metrics:
        show_metrics(agent, samples, ids, cats, products, args.dataset)
        return

    if args.sample:
        chosen = [s for s in samples if s["sample_id"] == args.sample]
        if not chosen:
            sys.exit("no sample named " + args.sample)
    elif args.scenario:
        chosen = [s for s in samples if s["scenario_type"] == args.scenario][:1]
    else:
        chosen = [next(s for s in samples if s["scenario_type"] == t)
                  for t in ("buying", "browsing", "intent_override", "boundary")]

    for s in chosen:
        run_session(agent, s, ids, cats, products)


if __name__ == "__main__":
    main()
