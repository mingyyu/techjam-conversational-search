"""Fast tests. No catalog download required except for the integration test,
which builds a tiny synthetic catalog on the fly.

Run:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.dialog import DialogState, WEIGHT_HARD, WEIGHT_SOFT
from src.routing import IntentRouter, TRACKS, BUYING, BROWSING
from src.catalog import coarse_category, normalise_phrase, attribute_phrases
from src.shopping_agent import ShoppingAgent, ATTRIBUTES


def state() -> DialogState:
    return DialogState(session_id="t", profile={})


class TestDialogParsing(unittest.TestCase):
    def test_buying_opening_extracts_category_and_requirement(self):
        s = state()
        s.ingest("I'm looking for Tops Tees & Blouses. A key requirement is: 100% cotton.")
        self.assertEqual(s.category, "Tops Tees & Blouses")
        self.assertEqual(len(s.constraints), 1)
        self.assertEqual(s.constraints[0].weight, WEIGHT_HARD)
        self.assertFalse(s.expects_override)

    def test_browsing_opening_has_category_but_no_constraint(self):
        s = state()
        s.ingest("I'm looking for Running Shoes, but I'm still exploring.")
        self.assertEqual(s.category, "Running Shoes")
        self.assertEqual(s.constraints, [])
        self.assertFalse(s.expects_override)

    def test_soft_opening_flags_a_pending_override(self):
        s = state()
        s.ingest("I'm looking for Wallets. I prefer a slim leather finish.")
        self.assertTrue(s.expects_override)
        self.assertEqual(s.constraints[0].weight, WEIGHT_SOFT)
        self.assertTrue(s.constraints[0].revocable)

    def test_disclosure_adds_each_semicolon_separated_constraint(self):
        s = state()
        s.ingest("I'm looking for Hats, but I'm still exploring.")
        s.ingest("For that, what matters is: merino wool; packable design.")
        self.assertEqual(len(s.constraints), 2)

    def test_declined_attribute_is_recorded_and_not_reasked(self):
        s = state()
        s.ingest("I don't have a preference for material; please use your judgment.")
        self.assertIn("material", s.dead_attributes)


class TestOverride(unittest.TestCase):
    def test_override_erases_revocable_and_asserts_replacement(self):
        s = state()
        s.ingest("I'm looking for Jackets. I prefer something lightweight.")
        self.assertEqual(len(s.constraints), 1)
        s.ingest("Actually, ignore my earlier preference. What I need is: waterproof shell.")
        self.assertTrue(s.override_seen)
        self.assertEqual(len(s.constraints), 1)
        self.assertEqual(s.constraints[0].text, "waterproof shell")
        self.assertEqual(s.constraints[0].weight, WEIGHT_HARD)

    def test_override_does_not_erase_hard_constraints(self):
        s = state()
        s.ingest("I'm looking for Boots. A key requirement is: steel toe.")
        s.ingest("Actually, ignore my earlier preference. What I need is: slip resistant.")
        texts = {c.text for c in s.constraints}
        self.assertIn("steel toe", texts)
        self.assertIn("slip resistant", texts)


class TestEliminationSafety(unittest.TestCase):
    """The regression this project's biggest bug produced.

    A product shown on a turn that did not convert is normally proof it is not
    the target. That inference is invalid while an override is pending, because
    the protocol suppresses conversion until the new intent arrives.
    """

    def test_eliminations_valid_in_ordinary_session(self):
        s = state()
        s.ingest("I'm looking for Socks. A key requirement is: merino.")
        self.assertTrue(s.eliminations_are_valid())

    def test_eliminations_suspended_while_override_pending(self):
        s = state()
        s.ingest("I'm looking for Socks. I like bright colours.")
        self.assertFalse(s.eliminations_are_valid())

    def test_eliminations_resume_after_override_lands(self):
        s = state()
        s.ingest("I'm looking for Socks. I like bright colours.")
        s.ingest("Actually, ignore my earlier preference. What I need is: merino.")
        self.assertTrue(s.eliminations_are_valid())


class TestCatalogHelpers(unittest.TestCase):
    def test_coarse_category_drops_generic_department(self):
        got = coarse_category(["Clothing, Shoes & Jewelry", "Women", "Tops, Tees & Blouses"])
        self.assertNotIn("Clothing,", got)
        self.assertIn("Tees & Blouses", got)

    def test_normalise_phrase_is_stable_under_whitespace_and_case(self):
        self.assertEqual(normalise_phrase("  100%   COTTON. "), normalise_phrase("100% cotton"))

    def test_attribute_phrases_include_material_and_budget(self):
        product = {"title": "Tee", "features": ["Soft cotton blend"], "price": 19.99}
        phrases = attribute_phrases(product)
        self.assertTrue(any("cotton" in p for p in phrases))
        self.assertTrue(any("budget around" in p for p in phrases))


class TestOutputContract(unittest.TestCase):
    """The harness treats malformed output as a miss, so this is load-bearing."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "catalog.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for i in range(40):
                fh.write(json.dumps({
                    "parent_asin": f"B{i:09d}",
                    "title": f"Sample cotton shirt number {i}",
                    "features": [f"feature variant {i}", "Soft cotton blend"],
                    "details": {"Fit": "regular"},
                    "description": ["A shirt."],
                    "categories": ["Clothing, Shoes & Jewelry", "Men", "Shirts"],
                    "price": 20.0 + i,
                    "average_rating": 4.5,
                    "rating_number": 100 + i,
                    "store": "SampleStore",
                }) + "\n")
        cls.agent = ShoppingAgent(str(path))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_response_shape_is_valid(self):
        self.agent.reset("s1", {"preference_tags": ["fit"]})
        out = self.agent.respond("s1", "I'm looking for Men Shirts. A key requirement is: Soft cotton blend.", 1, 10)
        self.assertIsInstance(out["message"], str)
        self.assertTrue(out["ask_attribute"] is None or out["ask_attribute"] in ATTRIBUTES)
        self.assertLessEqual(len(out["recommendations"]), 10)
        self.assertTrue(all("parent_asin" in r for r in out["recommendations"]))
        self.assertGreaterEqual(out["usage"]["prompt_tokens"], 0)

    def test_recommendations_never_repeat_across_turns(self):
        self.agent.reset("s2", {})
        seen: set[str] = set()
        for turn in range(1, 5):
            out = self.agent.respond("s2", "For that, what matters is: regular fit.", turn, 10)
            ids = [r["parent_asin"] for r in out["recommendations"]]
            self.assertEqual(len(ids), len(set(ids)), "duplicates within one turn")
            self.assertFalse(seen & set(ids), "product repeated across turns")
            seen.update(ids)

    def test_always_returns_a_full_list_even_with_no_information(self):
        self.agent.reset("s3", {})
        out = self.agent.respond("s3", "", 1, 10)
        self.assertEqual(len(out["recommendations"]), 10)

    def test_unknown_session_id_is_handled_without_raising(self):
        out = self.agent.respond("never-seen", "I'm looking for Men Shirts.", 1, 10)
        self.assertIsInstance(out["recommendations"], list)


class TestProfileDistillation(unittest.TestCase):
    def test_tags_expand_to_catalog_vocabulary(self):
        from src.profile import distill
        d = distill({"preference_tags": ["warmth", "weather"], "rating_style": "critical",
                     "average_prior_rating": 4.0})
        self.assertIn("insulated", d.query_terms)
        self.assertIn("waterproof", d.query_terms)
        self.assertLess(d.popularity_trust, 1.0)

    def test_tag_order_drives_question_order(self):
        from src.profile import distill
        d = distill({"preference_tags": ["material", "fit"]})
        self.assertEqual(d.attribute_order[:2], ["material", "size"])

    def test_missing_profile_is_safe(self):
        from src.profile import distill
        d = distill(None)
        self.assertEqual(d.query_terms, [])
        self.assertIsNone(d.rating_target)


class TestOrchestration(unittest.TestCase):
    def test_starts_focused(self):
        from src.strategy import Orchestrator, FOCUS
        self.assertEqual(Orchestrator().observe(1, True, 500, 0, True), FOCUS)

    def test_stall_escalates_to_broaden(self):
        from src.strategy import Orchestrator, BROADEN
        o = Orchestrator()
        o.observe(1, True, 500, 0, True)
        o.observe(2, False, 500, 10, True)
        self.assertEqual(o.observe(3, False, 500, 20, True), BROADEN)

    def test_new_information_resets_the_stall_counter(self):
        from src.strategy import Orchestrator, FOCUS
        o = Orchestrator()
        o.observe(1, True, 500, 0, True)
        o.observe(2, False, 500, 10, True)
        o.observe(3, True, 500, 20, True)
        self.assertEqual(o.mode, FOCUS)

    def test_pool_exhaustion_forces_broaden(self):
        from src.strategy import Orchestrator, BROADEN
        o = Orchestrator()
        self.assertEqual(o.observe(2, True, 40, 30, True), BROADEN)

    def test_over_generality_is_flagged(self):
        from src.strategy import Orchestrator
        o = Orchestrator()
        o.observe(1, False, 9000, 0, False)
        self.assertTrue(o.over_general)

    def test_transitions_are_recorded(self):
        from src.strategy import Orchestrator
        o = Orchestrator()
        o.observe(1, True, 500, 0, True)
        o.observe(2, False, 500, 10, True)
        o.observe(3, False, 500, 20, True)
        self.assertTrue(any("->" in t for t in o.transitions))


if __name__ == "__main__":
    unittest.main()


class DualTrackRoutingTests(unittest.TestCase):
    """Pillar I: intent must be inferred, since the harness never supplies it."""

    def state(self):
        return DialogState(session_id="s", profile={})

    def test_stated_requirement_routes_to_buying(self):
        router = IntentRouter()
        st = self.state()
        st.ingest("I'm looking for Women Jeans. A key requirement is: 100% Cotton.")
        track = router.observe("I'm looking for Women Jeans. A key requirement is: "
                               "100% Cotton.", st, turn=1)
        self.assertEqual(track.name, BUYING)

    def test_open_ended_opening_routes_to_browsing(self):
        router = IntentRouter()
        st = self.state()
        msg = "I'm looking for Women Jeans, but I'm still exploring."
        st.ingest(msg)
        self.assertEqual(router.observe(msg, st, turn=1).name, BROWSING)

    def test_untemplated_requirement_still_routes_to_buying(self):
        """Detection must not depend on the evaluator's sentence shapes."""
        router = IntentRouter()
        st = self.state()
        msg = "i need jeans that are definitely 100% cotton, must be machine washable"
        st.ingest(msg)
        self.assertEqual(router.observe(msg, st, turn=1).name, BUYING)

    def test_browsing_then_requirement_switches_track(self):
        router = IntentRouter()
        st = self.state()
        first = "I'm looking for Women Jeans, but I'm still exploring."
        st.ingest(first)
        self.assertEqual(router.observe(first, st, turn=1).name, BROWSING)
        second = "For that, what matters is: 100% Cotton."
        st.ingest(second)
        self.assertEqual(router.observe(second, st, turn=2).name, BUYING)
        self.assertTrue(router.transitions)

    def test_buying_track_leans_harder_on_the_customer_s_own_words(self):
        """What the split actually buys, after measurement.

        Discounting the popularity prior on the buying track was the original
        design and was rejected on evidence: it helps only when targets are
        long-tail, and real purchases are not. The tracks now differ in how much
        weight the customer's own vocabulary carries.
        """
        self.assertGreater(TRACKS[BUYING].w_bm25, TRACKS[BROWSING].w_bm25)
        self.assertEqual(TRACKS[BUYING].w_popularity, TRACKS[BROWSING].w_popularity)

    def test_browsing_track_spreads_picks_and_buying_does_not(self):
        self.assertTrue(TRACKS[BROWSING].diversify_early)
        self.assertFalse(TRACKS[BUYING].diversify_early)
