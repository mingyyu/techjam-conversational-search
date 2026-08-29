"""LLM paraphrase stress test.

The submitted agent reproduces the evaluator's own string-generation functions
(`coarse_category`, `_flatten_values`, `_clean_constraint`) character-for-character,
so the customer's templated wording resolves to an exact category pool and an exact
phrase-index key. The organizer's private 800 sessions may not be templated.

This harness rewrites the simulated customer's messages with a real LLM before the
agent sees them, then re-scores with the unmodified official evaluator. Ground truth
and scoring are untouched -- only the agent's input changes.

Two modes:
  opening  -- paraphrase only the first message of each session (isolates the leak)
  all      -- paraphrase every customer utterance (full stress)

Paraphrases are cached to disk so repeat runs are free and deterministic.

    python llm_stress.py --mode opening --agent submission
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from evaluator.local_evaluator import (
    evaluate, load_jsonl, catalog_index, coarse_category,
)

CACHE_PATH = Path("reports/paraphrase_cache.json")
_cache_lock = threading.Lock()


def load_env(path: str = ".env") -> None:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


SYSTEM = (
    "You rewrite a shopper's message so it sounds like something a real person "
    "typed into a chat box. Keep every requirement and the product type intact. "
    "Change the wording, phrasing and sentence shape. Do not quote product-listing "
    "text verbatim. Do not add requirements that were not there. Do not add "
    "commentary. Output only the rewritten message, one or two sentences."
)


class Paraphraser:
    def __init__(self, model: str | None = None) -> None:
        load_env()
        self.base = os.environ["LLM_BASE_URL"].rstrip("/")
        self.key = os.environ["LLM_API_KEY"]
        self.model = model or os.environ.get("LLM_MODEL", "qwen3.8:27b")
        self._local = threading.local()
        self.cache: dict[str, str] = {}
        if CACHE_PATH.exists():
            self.cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.failures: list[str] = []

    @property
    def session(self) -> requests.Session:
        # requests.Session is not thread-safe; give each worker its own.
        s = getattr(self._local, "session", None)
        if s is None:
            s = self._local.session = requests.Session()
        return s

    def _key(self, message: str) -> str:
        return f"{self.model}\0{message}"

    def _call(self, message: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": message},
            ],
            # Qwen3 non-thinking ("instruct") settings, as specified.
            "temperature": 0.7,
            "top_p": 0.80,
            "presence_penalty": 1.5,
            "max_tokens": 160,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for attempt in range(4):
            try:
                r = self.session.post(
                    f"{self.base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json=body, timeout=120,
                )
                r.raise_for_status()
                data = r.json()
                text = (data["choices"][0]["message"]["content"] or "").strip()
                usage = data.get("usage") or {}
                with _cache_lock:
                    self.prompt_tokens += int(usage.get("prompt_tokens", 0))
                    self.completion_tokens += int(usage.get("completion_tokens", 0))
                    self.calls += 1
                text = text.strip().strip('"').strip()
                return text or message
            except Exception as exc:
                if attempt == 3:
                    with _cache_lock:
                        self.failures.append(repr(exc)[:200])
                    return message  # fail open: unparaphrased, not a crash
                time.sleep(1.5 * (attempt + 1))
        return message

    def get(self, message: str) -> str:
        if not message.strip():
            return message
        k = self._key(message)
        with _cache_lock:
            hit = self.cache.get(k)
        if hit is not None:
            return hit
        out = self._call(message)
        if out != message:            # never cache a fail-open passthrough
            with _cache_lock:
                self.cache[k] = out
        return out

    def warm(self, messages: list[str], workers: int = 12) -> None:
        todo = [m for m in dict.fromkeys(messages)
                if m.strip() and self._key(m) not in self.cache]
        if not todo:
            return
        print(f"  paraphrasing {len(todo)} messages with {self.model} ...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(self.get, todo))
        self.save()

    def save(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _cache_lock:
            CACHE_PATH.write_text(json.dumps(self.cache, indent=1), encoding="utf-8")


class LLMPerturbed:
    """Rewrites customer messages with the LLM before the agent sees them."""

    def __init__(self, inner, para: Paraphraser, mode: str) -> None:
        self.inner, self.para, self.mode = inner, para, mode

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._turn = 0
        self.inner.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self._turn += 1
        rewrite = self.mode == "all" or (self.mode == "opening" and turn == 1)
        message = self.para.get(user_message) if rewrite else user_message
        return self.inner.respond(session_id, message, turn, top_k)


# --------------------------------------------------------------------------
# Control agent: the two dataset leaks and nothing else. No language
# understanding, no constraint parsing, no retrieval.
# --------------------------------------------------------------------------
class ControlAgent:
    def __init__(self, products: dict) -> None:
        self.members: dict[str, list[str]] = defaultdict(list)
        self.pop: dict[str, float] = {}
        for asin, p in products.items():
            self.members[coarse_category(p.get("categories") or []).lower()].append(asin)
            self.pop[asin] = math.log1p(float(p.get("rating_number") or 0))
        self.by_pop = sorted(self.pop, key=lambda a: -self.pop[a])

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.shown: set[str] = set()
        self.pool: list[str] | None = None
        self.safe = True

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        low = user_message.lower()
        if "looking for" in low and "still exploring" not in low and "key requirement" not in low:
            self.safe = False
        if "ignore my earlier preference" in low:
            self.safe = True
        m = re.search(r"looking for (.+?)(?:,? but|\. |$)", user_message)
        if m and self.pool is None:
            key = m.group(1).strip().rstrip(".").lower()
            self.pool = sorted(self.members.get(key, []), key=lambda a: -self.pop[a])
        out: list[str] = []
        for source in (self.pool or self.by_pop, self.by_pop):
            for asin in source:
                if len(out) >= top_k:
                    break
                if asin in self.shown or asin in out:
                    continue
                out.append(asin)
                if self.safe:
                    self.shown.add(asin)
        return {"message": "Here are some options. Anything else matter?",
                "ask_attribute": "other",
                "recommendations": [{"parent_asin": a} for a in out]}


def build_agent(kind: str, products: dict):
    if kind == "control":
        return ControlAgent(products)
    from starter.agent import Agent
    return Agent("data/catalog.jsonl")


def row(name: str, result: dict) -> dict:
    return {
        "run": name,
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "score": result["recommended_technical_score"],
        "scenario": {k: v["hit_rate_at_10"] for k, v in result["scenario_metrics"].items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="submission", choices=["submission", "control"])
    ap.add_argument("--mode", default="opening", choices=["opening", "all", "clean"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    samples = load_jsonl("data/public_set.jsonl")
    ids, cats, products = catalog_index("data/catalog.jsonl")
    agent = build_agent(args.agent, products)

    if args.mode == "clean":
        result = evaluate(agent, samples, ids, cats, products)
        print(json.dumps(row(f"{args.agent}/clean", result), indent=2))
        return

    para = Paraphraser(args.model)

    if args.mode == "opening":
        # Opening messages are fully determined by the sample, so warm them all
        # in parallel before scoring.
        openings = []
        for s in samples:
            target = str(s["ground_truth"]["parent_asin"])
            category = coarse_category(cats.get(target, []))
            scenario = s["scenario_type"]
            if scenario == "buying":
                from evaluator.local_evaluator import intent_card
                card = intent_card(products[target])
                openings.append(
                    f"I'm looking for {category}. A key requirement is: {card['hard_constraints'][0]}.")
            elif scenario == "intent_override":
                from evaluator.local_evaluator import intent_card, behavior_for
                import random as _r
                card = intent_card(products[target])
                beh = behavior_for(scenario, card,
                                   _r.Random(f"{s.get('sample_id','')}\0{scenario}"))
                openings.append(f"I'm looking for {category}. {beh['override']['old_value']}")
            else:
                openings.append(f"I'm looking for {category}, but I'm still exploring.")
        para.warm(openings)

    wrapped = LLMPerturbed(agent, para, args.mode)
    result = evaluate(wrapped, samples, ids, cats, products)
    para.save()

    out = row(f"{args.agent}/{args.mode}", result)
    out["paraphrase_model"] = para.model
    out["paraphrase_calls_this_run"] = para.calls
    out["paraphrase_tokens"] = {"prompt": para.prompt_tokens,
                                "completion": para.completion_tokens}
    print(json.dumps(out, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
