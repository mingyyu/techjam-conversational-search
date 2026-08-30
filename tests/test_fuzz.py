"""Adversarial input fuzz against the output contract.

`docs/competition_specification.md` says exceptions, invalid output and timeouts
may each count as a miss, and a miss is the most expensive thing that can happen
to the score -- roughly 0.005 per session, against 0.0010 for a rank slip and
0.0001 for an extra turn. The agent is therefore not allowed to raise, to return
an identifier outside the catalog, or to stall, whatever arrives in
`user_message`. Nothing in the official simulator produces input like this; the
point is that the organizer's environment is not obliged to resemble it.

Uses the same synthetic catalog as test_agent.py, so no download is required.
"""

from __future__ import annotations

import json
import random
import tempfile
import time
import unittest
from pathlib import Path

from src.shopping_agent import ShoppingAgent, ATTRIBUTES

ALLOWED = set(ATTRIBUTES)

# Hand-picked shapes that have a reason to break something: empty and
# whitespace-only messages, a template prefix with its payload missing, the
# separators the disclosure parser splits on, encodings a cp1252 console cannot
# render, and payloads that would matter if any of this were ever interpolated
# into a query, a shell, or a page.
EDGE_CASES = [
    "", " ", "\n\t ", "\r\n\r\n",
    "a" * 20000,
    "\x00\x01\x02 control characters",
    "I'm looking for .",
    "I'm looking for , but I'm still exploring.",
    "For that, what matters is: ",
    "For that, what matters is: ;;;;;",
    "Actually, ignore my earlier preference. What I need is: ",
    "I don't have a preference for ;",
    "I don't have a preference for zzzznotanattribute;",
    "\U0001F9E5 \U0001F460 emoji only",
    "SELECT * FROM products; DROP TABLE catalog--",
    "<script>alert(1)</script>",
    '{"parent_asin": "B000000000"}',
    "../../etc/passwd",
    "中文 テスト 한국어",
    "%s %d %n {0} {}",
    "-" * 500,
    "; ".join(["x"] * 300),
    "I'm looking for " + "z" * 5000 + ".",
    "NaN Infinity -0",
]


def random_cases(count: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    return ["".join(chr(rng.randrange(32, 0x2000))
                    for _ in range(rng.randrange(1, 300)))
            for _ in range(count)]


class TestAdversarialInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "catalog.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for i in range(40):
                handle.write(json.dumps({
                    "parent_asin": f"B{i:09d}",
                    "title": f"Sample cotton shirt number {i}",
                    "features": [f"feature variant {i}", "Soft cotton blend"],
                    "details": {"Fit": "regular"},
                    "description": ["A shirt."],
                    "categories": ["Clothing, Shoes & Jewelry", "Men", "Shirts"],
                    "price": 20.0 + i,
                    "average_rating": 4.5,
                    "rating_number": 100 + i,
                    "store": f"Store {i % 5}",
                }) + "\n")
        cls.agent = ShoppingAgent(str(path))
        cls.ids = {f"B{i:09d}" for i in range(40)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def assert_contract(self, response, message):
        note = repr(message[:60])
        self.assertIsInstance(response, dict, note)
        self.assertIsInstance(response["message"], str, note)

        attribute = response["ask_attribute"]
        self.assertTrue(attribute is None or attribute in ALLOWED,
                        f"{note}: illegal ask_attribute {attribute!r}")

        recommendations = response["recommendations"]
        self.assertIsInstance(recommendations, list, note)
        self.assertLessEqual(len(recommendations), 10, note)

        seen = set()
        for item in recommendations:
            asin = item["parent_asin"]
            self.assertIn(asin, self.ids, f"{note}: identifier outside the catalog")
            self.assertNotIn(asin, seen, f"{note}: duplicate identifier")
            seen.add(asin)

        usage = response["usage"]
        self.assertGreaterEqual(usage["prompt_tokens"], 0, note)
        self.assertGreaterEqual(usage["completion_tokens"], 0, note)

    def test_hostile_messages_never_break_the_contract(self):
        for i, message in enumerate(EDGE_CASES + random_cases(60)):
            session = f"fuzz-{i % 7}"
            if i % 7 == 0:
                self.agent.reset(session, {"preference_tags": ["fit"],
                                           "rating_style": "mixed"})
            self.assert_contract(
                self.agent.respond(session, message, (i % 10) + 1, 10), message)

    def test_no_single_turn_stalls(self):
        """A timeout may count as a miss, so pathological input must stay fast."""
        self.agent.reset("slow", {})
        for i, message in enumerate(EDGE_CASES, 1):
            start = time.perf_counter()
            self.agent.respond("slow", message, min(i, 10), 10)
            elapsed = time.perf_counter() - start
            self.assertLess(elapsed, 5.0,
                            f"{message[:60]!r} took {elapsed:.1f}s")

    def test_respond_without_reset_is_safe(self):
        fresh = ShoppingAgent.__new__(ShoppingAgent)
        fresh.index = self.agent.index
        fresh.state = None
        fresh._bm25_cache = {}
        self.assert_contract(
            fresh.respond("never-reset", "I'm looking for Men Shirts.", 1, 10),
            "no reset")

    def test_turn_numbers_outside_the_protocol_are_survivable(self):
        self.agent.reset("odd", {})
        for turn in (0, 1, 10, 99, -1):
            self.assert_contract(
                self.agent.respond("odd", "For that, what matters is: cotton.",
                                   turn, 10), f"turn={turn}")
