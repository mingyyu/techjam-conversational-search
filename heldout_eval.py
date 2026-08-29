"""Held-out evaluation harness.

Replays evaluation offline by rewriting customer messages before the agent
sees them, using pre-generated held-out perturbation dictionaries.

Usage:
    python heldout_eval.py --rewrites reports/heldout_natural.json --dataset data/public_set.jsonl
    python heldout_eval.py --rewrites reports/heldout_terse.json --dataset data/heldout_sessions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
from starter.agent import Agent


class PerturbedAgent:
    """Wraps an Agent and rewrites incoming customer messages using a lookup table."""

    def __init__(self, inner, rewrites: dict[str, str] | None = None) -> None:
        self.inner = inner
        self.rewrites = rewrites or {}
        self.rewritten_count = 0
        self.total_messages_seen = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.inner.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.total_messages_seen += 1
        msg = user_message or ""
        if msg in self.rewrites:
            self.rewritten_count += 1
            rewritten = self.rewrites[msg]
        else:
            rewritten = msg
        return self.inner.respond(session_id, rewritten, turn, top_k)


def run_evaluation(
    catalog_path: str = "data/catalog.jsonl",
    dataset_path: str = "data/public_set.jsonl",
    rewrites_path: str | None = None,
    output_path: str | None = None,
    agent_instance=None,
) -> dict:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)

    rewrites = {}
    if rewrites_path:
        with Path(rewrites_path).open("r", encoding="utf-8") as f:
            rewrites = json.load(f)

    base_agent = agent_instance or Agent(catalog_path)
    wrapped_agent = PerturbedAgent(base_agent, rewrites)

    result = evaluate(wrapped_agent, samples, catalog_ids, categories, products)

    # Attach harness metadata
    result["harness"] = {
        "dataset": dataset_path,
        "sample_count": len(samples),
        "rewrites_file": rewrites_path,
        "rewrites_dict_size": len(rewrites),
        "messages_intercepted": wrapped_agent.total_messages_seen,
        "messages_rewritten": wrapped_agent.rewritten_count,
        "rewrite_coverage_ratio": (
            round(wrapped_agent.rewritten_count / wrapped_agent.total_messages_seen, 4)
            if wrapped_agent.total_messages_seen > 0
            else 1.0
        ),
    }

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def print_summary(result: dict) -> None:
    harness = result.get("harness", {})
    rewrites_file = harness.get("rewrites_file") or "None (clean baseline)"
    print("\n" + "=" * 70)
    print("Held-Out Evaluation Summary")
    print(f"Dataset:  {harness.get('dataset')} ({result['sample_count']} sessions)")
    print(f"Rewrites: {rewrites_file}")
    if harness.get("messages_intercepted", 0) > 0:
        print(
            f"Coverage: {harness.get('messages_rewritten')}/{harness.get('messages_intercepted')} "
            f"messages rewritten ({harness.get('rewrite_coverage_ratio', 0.0) * 100:.1f}%)"
        )
    print("-" * 70)
    print(
        f"Overall:  HitRate@10 = {result['hit_rate_at_10']:.4f} | "
        f"MRR = {result['mrr']:.4f} | "
        f"MTTC = {result['mttc']:.2f} | "
        f"Score = {result['recommended_technical_score']:.4f}"
    )
    print("-" * 70)
    print("Scenario Breakdown:")
    for sc_name, sc_metrics in result.get("scenario_metrics", {}).items():
        print(
            f"  {sc_name:<16}: N={sc_metrics['sample_count']:<3} | "
            f"HitRate@10={sc_metrics['hit_rate_at_10']:.4f} | "
            f"MRR={sc_metrics['mrr']:.4f} | "
            f"MTTC={sc_metrics['mttc']:.2f}"
        )
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-Out Robustness Evaluator")
    parser.add_argument("--catalog", default="data/catalog.jsonl", help="Path to catalog JSONL")
    parser.add_argument("--dataset", default="data/public_set.jsonl", help="Path to dataset JSONL")
    parser.add_argument("--rewrites", default=None, help="Path to JSON file containing message rewrites")
    parser.add_argument("--output", default=None, help="Optional path to write JSON evaluation results")
    args = parser.parse_args()

    result = run_evaluation(
        catalog_path=args.catalog,
        dataset_path=args.dataset,
        rewrites_path=args.rewrites,
        output_path=args.output,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
