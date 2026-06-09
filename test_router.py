"""
test_router.py
--------------
Test suite for query_router.py.
Zero external dependencies — pure Python unittest.

Run:
    python -m unittest test_router -v
    python -m pytest test_router.py -v
"""

from __future__ import annotations

import unittest
from typing import Dict, List

from context_window_engine import GroundTruth, QueryError

from query_router import (
    IntentResult,
    ParsedQuery,
    QueryRouter,
    RetrievalSignal,
    Route,
    RouterResult,
    classify_intent,
    parse_query,
)


# ---------------------------------------------------------------------------
# Synthetic dataset — same structure as test_engine.py
# ---------------------------------------------------------------------------

def _make_rows(n: int = 1000) -> List[Dict[str, str]]:
    """
    Deterministic dataset with known ground truth.
      i in [0,   600): category=A, amt=100.00  → $60,000
      i in [600, 900): category=B, amt=200.00  → $60,000
      i in [900,1000): category=C, amt=300.00  → $30,000
      Total: $150,000
      Gender alternates M/F → 500 each.
      State  alternates NY/CA → 500 each.
      is_fraud=1 every 10th row → 100 rows.
    """
    rows = []
    for i in range(n):
        if i < int(n * 0.6):
            cat, amt = "A", "100.00"
        elif i < int(n * 0.9):
            cat, amt = "B", "200.00"
        else:
            cat, amt = "C", "300.00"
        rows.append({
            "category": cat,
            "amt":      amt,
            "gender":   "F" if i % 2 == 0 else "M",
            "state":    "NY" if i % 2 == 0 else "CA",
            "is_fraud": "1" if i % 10 == 0 else "0",
            "merchant": f"merchant_{i}",
            "city":     "TestCity",
        })
    return rows


ROWS = _make_rows(1000)


# ---------------------------------------------------------------------------
# TestClassifyIntent — 28 tests
# ---------------------------------------------------------------------------

class TestClassifyIntent(unittest.TestCase):
    """
    Tests for classify_intent().
    Verifies that every canonical query type routes correctly,
    and that the tier and confidence signals are sensible.
    """

    # ── COMPUTATION — Tier 1 (aggregation verbs) ─────────────────────────

    def test_total_spend_by_category(self):
        r = classify_intent("What is the total spend by category?")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 1)
        self.assertGreater(r.confidence, 0.9)

    def test_highest_average(self):
        r = classify_intent("Which category has the highest average transaction amount?")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 1)

    def test_how_many_female(self):
        r = classify_intent("How many transactions were made by female customers?")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 1)

    def test_total_amount_on_grocery(self):
        r = classify_intent("What is the total amount spent on grocery_pos?")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 1)

    def test_lowest_state(self):
        r = classify_intent("Which state has the lowest total spending?")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 1)

    def test_percentage_fraudulent(self):
        r = classify_intent("What percentage of transactions are fraudulent?")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 1)

    def test_sum_keyword(self):
        r = classify_intent("Sum all transactions in category B")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 1)

    def test_count_keyword(self):
        r = classify_intent("Count transactions where gender is F")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 1)

    def test_average_keyword(self):
        r = classify_intent("What is the average transaction amount?")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 1)

    def test_breakdown_keyword(self):
        r = classify_intent("Give me a breakdown of spend by state")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 1)

    def test_distribution_keyword(self):
        r = classify_intent("Show the distribution of transaction amounts")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 1)

    def test_rank_keyword(self):
        r = classify_intent("Rank categories by total spend")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 1)

    def test_top_n(self):
        r = classify_intent("Top 5 categories by spending")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 1)

    # ── COMPUTATION — Tier 2 (numeric comparison) ─────────────────────────

    def test_greater_than(self):
        r = classify_intent("Total spend where amount is greater than 500")
        self.assertEqual(r.route, Route.COMPUTATION)
        # may be tier 1 (total) or tier 2 — either is correct
        self.assertIn(r.matched_tier, (1, 2))

    def test_above_threshold(self):
        r = classify_intent("Transactions above $1000")
        self.assertEqual(r.route, Route.COMPUTATION)

    def test_less_than(self):
        r = classify_intent("Transactions less than 50")
        self.assertEqual(r.route, Route.COMPUTATION)

    def test_at_least(self):
        r = classify_intent("Spend at least 200")
        self.assertEqual(r.route, Route.COMPUTATION)

    def test_dollar_sign_trigger(self):
        r = classify_intent("All rows with $500")
        self.assertEqual(r.route, Route.COMPUTATION)

    # ── RETRIEVAL — Tier 3 ────────────────────────────────────────────────

    def test_find_lookup(self):
        r = classify_intent("Find transactions from Jennifer Banks")
        self.assertEqual(r.route, Route.RETRIEVAL)
        self.assertEqual(r.matched_tier, 3)

    def test_show_me(self):
        r = classify_intent("Show me a sample transaction from Texas")
        self.assertEqual(r.route, Route.RETRIEVAL)
        self.assertEqual(r.matched_tier, 3)

    def test_list_keyword(self):
        r = classify_intent("List all merchants in California")
        self.assertEqual(r.route, Route.RETRIEVAL)
        self.assertEqual(r.matched_tier, 3)

    def test_fetch_keyword(self):
        r = classify_intent("Fetch the latest transaction")
        self.assertEqual(r.route, Route.RETRIEVAL)
        self.assertEqual(r.matched_tier, 3)

    def test_search_keyword(self):
        r = classify_intent("Search for fraud transactions in NY")
        self.assertEqual(r.route, Route.RETRIEVAL)
        self.assertEqual(r.matched_tier, 3)

    # ── Structural guarantees ─────────────────────────────────────────────

    def test_returns_intent_result(self):
        r = classify_intent("What is the total spend?")
        self.assertIsInstance(r, IntentResult)

    def test_confidence_bounded(self):
        for q in [
            "total by category",
            "find Jennifer",
            "above $500",
            "",
            "xyz abc",
        ]:
            r = classify_intent(q)
            self.assertGreaterEqual(r.confidence, 0.0)
            self.assertLessEqual(r.confidence, 1.0)

    def test_empty_query_defaults_computation(self):
        r = classify_intent("")
        self.assertEqual(r.route, Route.COMPUTATION)

    def test_unknown_query_defaults_computation(self):
        # No pattern matches — should default to COMPUTATION (safer)
        r = classify_intent("xyzzy plugh foo bar baz")
        self.assertEqual(r.route, Route.COMPUTATION)
        self.assertEqual(r.matched_tier, 0)

    def test_reason_is_nonempty(self):
        for q in ["total spend", "find record", ""]:
            r = classify_intent(q)
            self.assertIsInstance(r.reason, str)
            self.assertGreater(len(r.reason), 0)


# ---------------------------------------------------------------------------
# TestParseQuery — 16 tests
# ---------------------------------------------------------------------------

class TestParseQuery(unittest.TestCase):
    """
    Tests for parse_query().
    Verifies that natural language maps to correct engine parameters.
    """

    def test_total_by_category(self):
        p = parse_query("What is the total spend by category?")
        self.assertEqual(p.agg_func, "sum")
        self.assertEqual(p.group_col, "category")
        self.assertIsNone(p.filter_col)

    def test_average_by_category(self):
        p = parse_query("What is the average transaction amount by category?")
        self.assertEqual(p.agg_func, "avg")
        self.assertEqual(p.group_col, "category")

    def test_filter_sum_grocery(self):
        p = parse_query("What is the total amount spent on grocery_pos?")
        self.assertEqual(p.agg_func, "sum")
        self.assertEqual(p.filter_col, "category")
        self.assertEqual(p.filter_val, "grocery_pos")

    def test_count_female(self):
        p = parse_query("How many transactions were made by female customers?")
        self.assertEqual(p.agg_func, "count")
        self.assertEqual(p.filter_col, "gender")
        self.assertEqual(p.filter_val, "F")

    def test_count_male(self):
        p = parse_query("How many transactions for male customers?")
        self.assertEqual(p.agg_func, "count")
        self.assertEqual(p.filter_col, "gender")
        self.assertEqual(p.filter_val, "M")

    def test_numeric_filter_gt(self):
        p = parse_query("Total spend where amount is greater than 500")
        self.assertEqual(p.agg_func, "sum")
        self.assertEqual(p.numeric_filter_col, "amt")
        self.assertEqual(p.numeric_filter_op, "gt")
        self.assertAlmostEqual(p.numeric_filter_val, 500.0)

    def test_numeric_filter_above(self):
        p = parse_query("Total spend above 1000")
        self.assertEqual(p.numeric_filter_op, "gt")
        self.assertAlmostEqual(p.numeric_filter_val, 1000.0)

    def test_numeric_filter_less_than(self):
        p = parse_query("Total transactions less than 100")
        self.assertEqual(p.numeric_filter_op, "lt")
        self.assertAlmostEqual(p.numeric_filter_val, 100.0)

    def test_numeric_filter_at_least(self):
        p = parse_query("Total spend at least 250")
        self.assertEqual(p.numeric_filter_op, "gte")
        self.assertAlmostEqual(p.numeric_filter_val, 250.0)

    def test_lowest_state(self):
        p = parse_query("Which state has the lowest total spending?")
        self.assertEqual(p.group_col, "state")
        self.assertEqual(p.agg_func, "sum")

    def test_fraud_ratio(self):
        p = parse_query("What percentage of transactions are fraudulent?")
        # ratio queries map to SUM(is_fraud)
        self.assertEqual(p.agg_col, "is_fraud")
        self.assertEqual(p.agg_func, "sum")

    def test_group_by_state(self):
        p = parse_query("Total spend by state")
        self.assertEqual(p.group_col, "state")
        self.assertEqual(p.agg_func, "sum")

    def test_returns_parsed_query(self):
        p = parse_query("Total spend by category")
        self.assertIsInstance(p, ParsedQuery)

    def test_parse_confidence_bounded(self):
        for q in ["total by category", "find record", "", "unknown xyz"]:
            p = parse_query(q)
            self.assertGreaterEqual(p.parse_confidence, 0.0)
            self.assertLessEqual(p.parse_confidence, 1.0)

    def test_parse_notes_is_string(self):
        p = parse_query("Total spend by category")
        self.assertIsInstance(p.parse_notes, str)

    def test_dollar_amount_parsed(self):
        p = parse_query("Spend greater than $500")
        self.assertAlmostEqual(p.numeric_filter_val, 500.0)


# ---------------------------------------------------------------------------
# TestQueryRouter — 14 tests
# ---------------------------------------------------------------------------

class TestQueryRouter(unittest.TestCase):
    """
    Integration tests for QueryRouter against the synthetic dataset.
    Verifies routing decisions and answer correctness.
    """

    def setUp(self):
        self.router = QueryRouter(ROWS)

    def test_returns_router_result(self):
        r = self.router.route("What is the total spend by category?")
        self.assertIsInstance(r, RouterResult)

    def test_computation_routes_correctly(self):
        r = self.router.route("What is the total spend by category?")
        self.assertEqual(r.routed_to, Route.COMPUTATION)

    def test_retrieval_routes_correctly(self):
        r = self.router.route("Find transactions from merchant_1")
        self.assertEqual(r.routed_to, Route.RETRIEVAL)

    def test_computation_returns_ground_truth(self):
        r = self.router.route("What is the total spend?")
        self.assertIsInstance(r.answer, GroundTruth)

    def test_retrieval_returns_retrieval_signal(self):
        r = self.router.route("Find transactions from merchant_1")
        self.assertIsInstance(r.answer, RetrievalSignal)

    def test_total_sum_correct(self):
        # Dataset total = $150,000
        r = self.router.route("What is the total spend?")
        self.assertIsInstance(r.answer, GroundTruth)
        self.assertAlmostEqual(r.answer.answer, 150_000.0, delta=1.0)

    def test_group_by_category_correct(self):
        r = self.router.route("What is the total spend by category?")
        gt = r.answer
        self.assertIsInstance(gt, GroundTruth)
        answers = dict(gt.answer)
        self.assertAlmostEqual(answers.get("A", 0), 60_000.0, delta=1.0)
        self.assertAlmostEqual(answers.get("B", 0), 60_000.0, delta=1.0)
        self.assertAlmostEqual(answers.get("C", 0), 30_000.0, delta=1.0)

    def test_count_female_correct(self):
        r = self.router.route("How many transactions were made by female customers?")
        gt = r.answer
        self.assertIsInstance(gt, GroundTruth)
        self.assertAlmostEqual(gt.answer, 500.0, delta=1.0)

    def test_numeric_filter_gt(self):
        # Rows with amt > 150: all B (200) and C (300) rows = 400 rows
        r = self.router.route("Total spend where amount is greater than 150")
        gt = r.answer
        self.assertIsInstance(gt, GroundTruth)
        # B: 300 rows × $200 = $60,000; C: 100 rows × $300 = $30,000
        self.assertAlmostEqual(gt.answer, 90_000.0, delta=1.0)

    def test_latency_fields_populated(self):
        r = self.router.route("Total spend by category")
        self.assertGreater(r.route_latency, 0)
        self.assertGreater(r.total_latency, 0)
        self.assertGreaterEqual(r.total_latency, r.route_latency)

    def test_route_counts_increment(self):
        router = QueryRouter(ROWS)
        router.route("Total spend by category")    # COMPUTATION
        router.route("Find merchant_1")            # RETRIEVAL
        router.route("How many female customers")  # COMPUTATION
        counts = router.route_counts
        self.assertEqual(counts[Route.COMPUTATION], 2)
        self.assertEqual(counts[Route.RETRIEVAL],   1)

    def test_rag_warning_none_for_computation(self):
        r = self.router.route("Total spend by category")
        self.assertIsNone(r.rag_warning)

    def test_rag_warning_none_for_retrieval(self):
        # Retrieval is safe — no warning needed
        r = self.router.route("Find transactions from merchant_5")
        self.assertIsNone(r.rag_warning)

    def test_all_seven_benchmark_queries_route_computation(self):
        computation_queries = [
            "What is the total spend by category?",
            "Which category has the highest average transaction amount?",
            "What is the total amount spent on grocery_pos?",
            "How many transactions were made by female customers?",
            "What is the total spend where amount is greater than 500?",
            "Which state has the lowest total spending?",
            "What percentage of transactions are fraudulent?",
        ]
        for q in computation_queries:
            r = self.router.route(q)
            self.assertEqual(
                r.routed_to, Route.COMPUTATION,
                msg=f"Expected COMPUTATION for: '{q}', got {r.routed_to}"
            )


# ---------------------------------------------------------------------------
# TestRouterEdgeCases — 8 tests
# ---------------------------------------------------------------------------

class TestRouterEdgeCases(unittest.TestCase):

    def setUp(self):
        self.router = QueryRouter(ROWS)

    def test_empty_query(self):
        r = self.router.route("")
        # Empty query → defaults to COMPUTATION
        self.assertEqual(r.routed_to, Route.COMPUTATION)
        self.assertIsInstance(r, RouterResult)

    def test_single_word_query(self):
        r = self.router.route("total")
        self.assertEqual(r.routed_to, Route.COMPUTATION)

    def test_very_long_query(self):
        q = "What is the total spend by category " + "and also by state " * 20
        r = self.router.route(q)
        self.assertIsInstance(r, RouterResult)

    def test_all_caps_query(self):
        r = self.router.route("WHAT IS THE TOTAL SPEND BY CATEGORY")
        self.assertEqual(r.routed_to, Route.COMPUTATION)

    def test_query_with_numbers(self):
        r = self.router.route("Total spend greater than $1,000")
        self.assertEqual(r.routed_to, Route.COMPUTATION)

    def test_empty_dataset_router(self):
        router = QueryRouter([])
        r = router.route("Total spend by category")
        self.assertIsInstance(r, RouterResult)
        # answer should still be a GroundTruth (with 0 answer)
        self.assertIsInstance(r.answer, GroundTruth)

    def test_route_counts_start_zero(self):
        router = QueryRouter(ROWS)
        self.assertEqual(router.route_counts[Route.COMPUTATION], 0)
        self.assertEqual(router.route_counts[Route.RETRIEVAL],   0)

    def test_intent_result_properties(self):
        intent_comp = classify_intent("total spend by category")
        self.assertTrue(intent_comp.is_computation)
        self.assertFalse(intent_comp.is_retrieval)

        intent_retr = classify_intent("find transaction")
        self.assertTrue(intent_retr.is_retrieval)
        self.assertFalse(intent_retr.is_computation)


# ---------------------------------------------------------------------------
# TestRouterVsRAGContrast — 6 tests
# ---------------------------------------------------------------------------

class TestRouterVsRAGContrast(unittest.TestCase):
    """
    Verifies the core claim: for every aggregation query,
    the router returns the exact correct answer — the number
    that RAG cannot produce regardless of context size.
    """

    def setUp(self):
        self.router = QueryRouter(ROWS)

    def test_router_sum_is_exact(self):
        """Router answer equals independent ground truth — no approximation."""
        r = self.router.route("What is the total spend by category?")
        answers = dict(r.answer.answer)
        # A: 600 × 100 = 60000, B: 300 × 200 = 60000, C: 100 × 300 = 30000
        self.assertAlmostEqual(answers["A"], 60_000.0, delta=0.01)
        self.assertAlmostEqual(answers["B"], 60_000.0, delta=0.01)
        self.assertAlmostEqual(answers["C"], 30_000.0, delta=0.01)

    def test_router_count_is_exact(self):
        r = self.router.route("How many transactions were made by female customers?")
        self.assertAlmostEqual(r.answer.answer, 500.0, delta=0.01)

    def test_router_numeric_filter_is_exact(self):
        # amt > 100: B rows (200) + C rows (300) = 400 rows
        # total = 300×200 + 100×300 = 60000 + 30000 = 90000
        r = self.router.route("Total spend where amount is greater than 100")
        self.assertAlmostEqual(r.answer.answer, 90_000.0, delta=1.0)

    def test_router_scans_full_dataset(self):
        """Unlike RAG, the router always scans 100% of rows."""
        r = self.router.route("Total spend by category")
        gt = r.answer
        self.assertEqual(gt.rows_scanned, len(ROWS))

    def test_router_latency_is_measured(self):
        r = self.router.route("Total spend by category")
        self.assertGreater(r.answer.latency_ms, 0)

    def test_retrieval_signal_is_safe(self):
        """Retrieval signal correctly marks lookup queries as RAG-safe."""
        r = self.router.route("Find the transaction from merchant_5")
        self.assertIsInstance(r.answer, RetrievalSignal)
        self.assertTrue(r.answer.safe)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
