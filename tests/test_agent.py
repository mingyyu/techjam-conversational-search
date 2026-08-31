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

    def test_first_decline_is_a_deferral_not_an_exhausted_attribute(self):
        # The Boundary scenario refuses exactly one question per session and
        # still holds every constraint. Retiring the attribute here costs the
        # session its highest-yield question -- see DialogState.ingest.
        s = state()
        s.ingest("I don't have a preference for material; please use your judgment.")
        self.assertEqual(s.dead_attributes, set())

    def test_repeated_decline_is_recorded_and_not_reasked(self):
        s = state()
        s.ingest("I don't have a preference for material; please use your judgment.")
        s.ingest("I don't have an additional preference for material.")
        self.assertIn("material", s.dead_attributes)

    def test_boundary_deferral_keeps_the_open_question_alive(self):
        # The first question of a session is always `other`; the Boundary reply
        # must not retire it, because the constraints are still undisclosed.
        s = state()
        s.ingest("I'm looking for Shoes Slippers, but I'm still exploring.")
        s.ingest("I don't have a preference for other; please use your judgment.")
        self.assertNotIn("other", s.dead_attributes)
        s.ingest("For that, what matters is: memory foam; indoor outdoor sole.")
        self.assertEqual(len(s.constraints), 2)


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

    def test_commits_to_one_pick_while_the_customer_is_still_disclosing(self):
        from src.shopping_agent import COMMIT_TURNS, COMMIT_WIDTH
        self.agent.reset("s3", {})
        for turn in range(1, COMMIT_TURNS + 1):
            out = self.agent.respond("s3", "", turn, 10)
            # Never empty: a turn that recommends nothing cannot convert.
            self.assertEqual(len(out["recommendations"]), COMMIT_WIDTH)

    def test_returns_a_full_list_once_disclosures_are_exhausted(self):
        from src.shopping_agent import COMMIT_TURNS
        self.agent.reset("s4", {})
        for turn in range(1, COMMIT_TURNS + 1):
            self.agent.respond("s4", "", turn, 10)
        out = self.agent.respond("s4", "", COMMIT_TURNS + 1, 10)
        self.assertEqual(len(out["recommendations"]), 10)

    def test_never_exceeds_the_requested_top_k(self):
        self.agent.reset("s5", {})
        for turn in range(1, 6):
            out = self.agent.respond("s5", "", turn, 10)
            self.assertLessEqual(len(out["recommendations"]), 10)
            self.assertGreaterEqual(len(out["recommendations"]), 1)

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
        self.assertEqual(d.attribute_order, [])
        self.assertEqual(d.popularity_trust, 0.9)


class TestOrchestration(unittest.TestCase):
    def test_starts_focused(self):
        from src.strategy import Orchestrator, FOCUS
        self.assertEqual(Orchestrator().observe(1, True, 500, 0), FOCUS)

    def test_stall_escalates_to_broaden(self):
        from src.strategy import Orchestrator, BROADEN
        o = Orchestrator()
        o.observe(1, True, 500, 0)
        o.observe(2, False, 500, 10)
        self.assertEqual(o.observe(3, False, 500, 20), BROADEN)

    def test_new_information_resets_the_stall_counter(self):
        from src.strategy import Orchestrator, FOCUS
        o = Orchestrator()
        o.observe(1, True, 500, 0)
        o.observe(2, False, 500, 10)
        o.observe(3, True, 500, 20)
        self.assertEqual(o.mode, FOCUS)

    def test_pool_exhaustion_forces_broaden(self):
        from src.strategy import Orchestrator, BROADEN
        o = Orchestrator()
        self.assertEqual(o.observe(2, True, 40, 30), BROADEN)

    def test_transitions_are_recorded(self):
        from src.strategy import Orchestrator
        o = Orchestrator()
        o.observe(1, True, 500, 0)
        o.observe(2, False, 500, 10)
        o.observe(3, False, 500, 20)
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


class TestFreeFormRetraction(unittest.TestCase):
    """Retracting a subject in words a person would actually use.

    `TestOverride` above covers the templated form. This is the same act said
    conversationally, which until recently appended a constraint instead of
    clearing one -- so "never mind, show me shoes" kept ranking against coats.
    """

    def test_retraction_clears_salvaged_state(self):
        s = state()
        s.ingest("Hello! What is a good winter jacket")
        self.assertEqual(len(s.constraints), 1)
        self.assertTrue(s.free_text)
        s.ingest("okay never mind, gimme some nice sports shoes then")
        self.assertEqual(len(s.constraints), 1)
        self.assertNotIn("jacket", s.constraints[0].text)
        self.assertEqual(len(s.free_text), 1)

    def test_retraction_keeps_stated_requirements(self):
        s = state()
        s.ingest("I'm looking for Boots. A key requirement is: steel toe.")
        s.ingest("never mind the colour, show me something else")
        self.assertIn("steel toe", {c.text for c in s.constraints})

    def test_retraction_readmits_eliminated_products(self):
        s = state()
        s.ingest("a warm coat")
        s.shown.update({"A1", "A2"})
        s.ingest("forget that, I want running shoes")
        self.assertEqual(s.shown, set())

    def test_ordinary_paraphrase_is_not_a_retraction(self):
        """The expensive failure mode: a first draft matched "actually",
        "instead" and "that's not a ...", which are ordinary filler in reworded
        messages that are *adding* information. Wiping state on those cost 0.054
        on the natural set and 0.058 on the indirect set."""
        for message in ("actually I really need it waterproof",
                        "no, show me something warmer",
                        "I want a jacket, not a vest",
                        "instead of wool I would take fleece"):
            s = state()
            s.ingest("I'm looking for Jackets. A key requirement is: waterproof.")
            s.ingest(message)
            self.assertIn("waterproof", {c.text for c in s.constraints}, message)
            self.assertGreater(len(s.free_text), 0, message)


class TestOpenQuestionRetirement(unittest.TestCase):
    """The agent must stop asking "anything else?" when it stops working.

    Only the simulator declines in the exact words NO_PREFERENCE_RE matches, so
    against a real person the open question was re-asked every turn for ten
    turns and the attribute picker was never reached.
    """

    def test_open_question_retires_when_nothing_filterable_lands(self):
        from src.shopping_agent import OPEN_ASK_LIMIT
        s = state()
        for _ in range(OPEN_ASK_LIMIT):
            s.open_asks += 1
        self.assertTrue(ShoppingAgent._open_is_spent(s))

    def test_open_question_survives_while_the_session_is_landing_information(self):
        from src.shopping_agent import OPEN_ASK_LIMIT
        s = state()
        s.ingest("I'm looking for Boots. A key requirement is: steel toe.")
        s.open_asks = OPEN_ASK_LIMIT + 5
        self.assertFalse(ShoppingAgent._open_is_spent(s))


class TestSingularFolding(unittest.TestCase):
    def test_singular_folds_plurals_without_mangling_words(self):
        from src.catalog import singular
        self.assertEqual(singular("jackets"), "jacket")
        self.assertEqual(singular("shoes"), "shoe")
        self.assertEqual(singular("dress"), "dress")   # -ss is not a plural
        self.assertEqual(singular("gas"), "gas")       # too short to fold
