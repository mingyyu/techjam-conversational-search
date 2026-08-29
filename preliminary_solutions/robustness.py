"""Robustness harness.

The public set is 200 sessions of clean, templated customer language. The
private set is 800 sessions the organizer may have paraphrased. This script
perturbs the customer's wording in ways the private set plausibly differs and
reports what each perturbation costs.

Run from the competition kit root:

    python3 robustness.py

Nothing here modifies the evaluator; each stress case wraps the agent and
rewrites the message before it reaches `respond`.
"""

from __future__ import annotations

import json
import random
import re

from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
from starter.agent import Agent

# ---------------------------------------------------------------------------
# Paraphrase map. Hand-built and deliberately NOT derived from any model or
# from the catalog, so it cannot flatter a corpus-trained component.
# ---------------------------------------------------------------------------
PARAPHRASE = {
    "waterproof": "rainproof", "hiking": "trail walking", "warm": "cozy",
    "lightweight": "featherweight", "durable": "long lasting",
    "comfortable": "comfy", "sneakers": "trainers", "jacket": "coat",
    "pants": "trousers", "sweater": "jumper", "cotton": "natural fibre",
    "leather": "hide", "soft": "gentle", "stretch": "elastic",
    "breathable": "airy", "adjustable": "customisable",
    "pockets": "compartments", "zipper": "zip fastening", "casual": "everyday",
    "formal": "dressy", "running": "jogging", "winter": "cold weather",
    "summer": "hot weather", "large": "roomy", "small": "compact",
    "thick": "heavyweight", "thin": "slender", "bright": "vivid",
    "sleeve": "arm covering", "collar": "neckband", "hood": "head covering",
    "fit": "cut", "wool": "fleece wool", "polyester": "synthetic",
    "washable": "launderable", "quality": "craftsmanship",
}
PARAPHRASE_RE = re.compile(r"\b(" + "|".join(PARAPHRASE) + r")\b", re.I)

CATEGORY_RE = re.compile(r"(looking for\s+)(.+?)(\.|,\s*but\b)")


class Perturbed:
    """Rewrites customer messages before the agent sees them."""

    def __init__(self, inner, mode: str) -> None:
        self.inner = inner
        self.mode = mode

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.inner.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self.inner.respond(session_id, self._rewrite(user_message or ""), turn, top_k)

    def _rewrite(self, message: str) -> str:
        if self.mode == "paraphrase":
            # Requirement wording only; the category clause is left alone so
            # this isolates constraint vocabulary from routing.
            head, sep, tail = message.partition(".")
            return head + sep + PARAPHRASE_RE.sub(
                lambda m: PARAPHRASE[m.group(1).lower()], tail)

        match = CATEGORY_RE.search(message)
        if not match:
            return message
        category = match.group(2)
        rng = random.Random(category)

        if self.mode == "category_lowercase":
            replacement = category.lower()
        elif self.mode == "category_typo":
            replacement = "".join(
                c for i, c in enumerate(category) if i != len(category) // 2)
        elif self.mode == "category_drop_word":
            replacement = " ".join(category.split()[:-1]) or category
        elif self.mode == "category_reorder":
            words = category.split()
            rng.shuffle(words)
            replacement = " ".join(words)
        elif self.mode == "category_generic":
            replacement = "clothing item"
        else:
            replacement = category
        return message[:match.start(2)] + replacement + message[match.end(2):]


MODES = [
    ("none", "clean public-set wording"),
    ("paraphrase", "requirement vocabulary rewritten"),
    ("category_lowercase", "category casing changed"),
    ("category_typo", "one character dropped from the category"),
    ("category_reorder", "category words reordered"),
    ("category_drop_word", "last category word missing"),
    ("category_generic", "category replaced by a vague label"),
]


def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
    agent = Agent("data/catalog.jsonl")

    rows = []
    for mode, description in MODES:
        target = agent if mode == "none" else Perturbed(agent, mode)
        agent._impl._bm25_cache.clear()
        result = evaluate(target, samples, catalog_ids, categories, products)
        rows.append({
            "perturbation": mode,
            "description": description,
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "score": result["recommended_technical_score"],
        })
        print(f"{mode:<20} HR={result['hit_rate_at_10']:.3f} "
              f"MRR={result['mrr']:.4f} MTTC={result['mttc']:.3f} "
              f"score={result['recommended_technical_score']:.5f}", flush=True)

    with open("reports/robustness.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
