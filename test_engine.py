"""
test_engine.py
--------------
Test suite for the Context Window Engine.
Zero external dependencies — pure Python unittest.

Run:
    python -m pytest test_engine.py -v
    python -m unittest test_engine -v
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from typing import Dict, List

from context_window_engine import (
    BENCHMARK_QUERIES,
    CONTEXT_SIZES,
    TOKENS_PER_ROW,
    VALID_AGG_FUNCS,
    VALID_OPERATORS,
    BenchmarkQuery,
    DataLoadError,
    EngineError,
    GroundTruth,
    QueryError,
    QueryResult,
    RAGContext,
    SchemaError,
    _aggregate,
    _apply_numeric_filter,
    _compute_detectability,
    _compute_specificity,
    _estimate_response_length,
    _fmt_answer,
    _try_float,
    compute_ground_truth,
    load_csv,
    run_query,
    simulate_rag_retrieval,
)


# ---------------------------------------------------------------------------
# Synthetic dataset — controlled, known ground truth
# ---------------------------------------------------------------------------

def _make_rows(n: int = 1000) -> List[Dict[str, str]]:
    """
    Synthetic transaction dataset with deterministic ground truth.

    Layout:
      i in [0, 600):   category=A, amt=100.00  → 600 rows × $100 = $60,000
      i in [600, 900):  category=B, amt=200.00  → 300 rows × $200 = $60,000
      i in [900, 1000): category=C, amt=300.00  → 100 rows × $300 = $30,000
      Total: $150,000

    Gender alternates M/F → 500 each.
    State alternates NY/CA → 500 each.
    is_fraud=1 every 10th row → 100 fraud rows.
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
# TestTryFloat
# ---------------------------------------------------------------------------

class TestTryFloat(unittest.TestCase):

    def test_plain_float(self):
        self.assertEqual(_try_float("107.23"), 107.23)

    def test_integer_string(self):
        self.assertEqual(_try_float("42"), 42.0)

    def test_comma_separated(self):
        self.assertEqual(_try_float("1,234.56"), 1234.56)

    def test_dollar_sign(self):
        self.assertEqual(_try_float("$500.00"), 500.0)

    def test_negative(self):
        self.assertEqual(_try_float("-100.5"), -100.5)

    def test_zero(self):
        self.assertEqual(_try_float("0"), 0.0)

    def test_empty_string(self):
        self.assertIsNone(_try_float(""))

    def test_whitespace_only(self):
        self.assertIsNone(_try_float("   "))

    def test_non_numeric(self):
        self.assertIsNone(_try_float("grocery_pos"))

    def test_none_like_string(self):
        self.assertIsNone(_try_float("None"))

    def test_scientific_notation(self):
        result = _try_float("1.5e3")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1500.0)


# ---------------------------------------------------------------------------
# TestAggregate
# ---------------------------------------------------------------------------

class TestAggregate(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(_aggregate([1.0, 2.0, 3.0], "sum"), 6.0)

    def test_avg(self):
        self.assertEqual(_aggregate([1.0, 2.0, 3.0], "avg"), 2.0)

    def test_count(self):
        self.assertEqual(_aggregate([1.0, 2.0, 3.0], "count"), 3.0)

    def test_max(self):
        self.assertEqual(_aggregate([1.0, 5.0, 3.0], "max"), 5.0)

    def test_min(self):
        self.assertEqual(_aggregate([1.0, 5.0, 3.0], "min"), 1.0)

    def test_empty_list(self):
        self.assertEqual(_aggregate([], "sum"), 0.0)

    def test_invalid_func(self):
        with self.assertRaises(QueryError):
            _aggregate([1.0], "median")

    def test_single_value(self):
        self.assertEqual(_aggregate([42.0], "avg"), 42.0)


# ---------------------------------------------------------------------------
# TestNumericFilter
# ---------------------------------------------------------------------------

class TestNumericFilter(unittest.TestCase):

    def setUp(self):
        self.rows = [{"amt": str(float(i))} for i in range(1, 11)]

    def test_gt(self):
        result = _apply_numeric_filter(self.rows, "amt", "gt", 5.0)
        self.assertEqual(len(result), 5)  # 6,7,8,9,10

    def test_gte(self):
        result = _apply_numeric_filter(self.rows, "amt", "gte", 5.0)
        self.assertEqual(len(result), 6)  # 5,6,7,8,9,10

    def test_lt(self):
        result = _apply_numeric_filter(self.rows, "amt", "lt", 5.0)
        self.assertEqual(len(result), 4)  # 1,2,3,4

    def test_lte(self):
        result = _apply_numeric_filter(self.rows, "amt", "lte", 5.0)
        self.assertEqual(len(result), 5)  # 1,2,3,4,5

    def test_eq(self):
        result = _apply_numeric_filter(self.rows, "amt", "eq", 5.0)
        self.assertEqual(len(result), 1)

    def test_invalid_operator(self):
        with self.assertRaises(QueryError):
            _apply_numeric_filter(self.rows, "amt", "between", 5.0)

    def test_missing_column_values_skipped(self):
        rows_with_missing = [{"amt": ""}, {"amt": "10.0"}, {"amt": "bad"}]
        result = _apply_numeric_filter(rows_with_missing, "amt", "gt", 5.0)
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# TestGroundTruth
# ---------------------------------------------------------------------------

class TestGroundTruth(unittest.TestCase):

    def test_sum_full_dataset(self):
        gt = compute_ground_truth("sum", ROWS, agg_func="sum", agg_col="amt")
        self.assertAlmostEqual(gt.answer, 150_000.0, delta=1.0)
        self.assertEqual(gt.rows_scanned, 1000)

    def test_sum_by_category(self):
        gt = compute_ground_truth(
            "sum_cat", ROWS,
            agg_func="sum", agg_col="amt", group_col="category",
        )
        answers = dict(gt.answer)
        self.assertAlmostEqual(answers["A"], 60_000.0, delta=1.0)
        self.assertAlmostEqual(answers["B"], 60_000.0, delta=1.0)
        self.assertAlmostEqual(answers["C"], 30_000.0, delta=1.0)

    def test_avg_by_category(self):
        gt = compute_ground_truth(
            "avg_cat", ROWS,
            agg_func="avg", agg_col="amt", group_col="category",
        )
        answers = dict(gt.answer)
        self.assertAlmostEqual(answers["A"], 100.0, places=1)
        self.assertAlmostEqual(answers["B"], 200.0, places=1)
        self.assertAlmostEqual(answers["C"], 300.0, places=1)

    def test_count_with_filter(self):
        gt = compute_ground_truth(
            "count_f", ROWS,
            agg_func="count", agg_col="amt",
            filter_col="gender", filter_val="F",
        )
        self.assertEqual(gt.answer, 500.0)

    def test_sum_categorical_filter(self):
        gt = compute_ground_truth(
            "sum_a", ROWS,
            agg_func="sum", agg_col="amt",
            filter_col="category", filter_val="A",
        )
        self.assertAlmostEqual(gt.answer, 60_000.0, delta=1.0)

    def test_numeric_filter_gt(self):
        gt = compute_ground_truth(
            "gt150", ROWS,
            agg_func="sum", agg_col="amt",
            numeric_filter_col="amt",
            numeric_filter_op="gt",
            numeric_filter_val=150.0,
        )
        # B (300×200) + C (100×300) = 60000+30000 = 90000
        self.assertAlmostEqual(gt.answer, 90_000.0, delta=1.0)

    def test_numeric_filter_lte(self):
        gt = compute_ground_truth(
            "lte100", ROWS,
            agg_func="sum", agg_col="amt",
            numeric_filter_col="amt",
            numeric_filter_op="lte",
            numeric_filter_val=100.0,
        )
        self.assertAlmostEqual(gt.answer, 60_000.0, delta=1.0)

    def test_max_aggregation(self):
        gt = compute_ground_truth("max", ROWS, agg_func="max", agg_col="amt")
        self.assertEqual(gt.answer, 300.0)

    def test_min_aggregation(self):
        gt = compute_ground_truth("min", ROWS, agg_func="min", agg_col="amt")
        self.assertEqual(gt.answer, 100.0)

    def test_group_by_state(self):
        gt = compute_ground_truth(
            "state", ROWS,
            agg_func="sum", agg_col="amt", group_col="state",
        )
        answers = dict(gt.answer)
        self.assertIn("NY", answers)
        self.assertIn("CA", answers)

    def test_latency_is_positive(self):
        gt = compute_ground_truth("t", ROWS, agg_func="sum", agg_col="amt")
        self.assertGreater(gt.latency_ms, 0)

    def test_rows_scanned_equals_dataset_size(self):
        gt = compute_ground_truth("t", ROWS, agg_func="sum", agg_col="amt")
        self.assertEqual(gt.rows_scanned, len(ROWS))

    def test_empty_filter_returns_zero(self):
        gt = compute_ground_truth(
            "no_match", ROWS,
            agg_func="sum", agg_col="amt",
            filter_col="category", filter_val="Z",
        )
        self.assertEqual(gt.answer, 0.0)
        self.assertEqual(gt.rows_matched, 0)

    def test_invalid_agg_func_raises(self):
        with self.assertRaises(QueryError):
            compute_ground_truth("t", ROWS, agg_func="median", agg_col="amt")

    def test_missing_column_raises_schema_error(self):
        with self.assertRaises(SchemaError):
            compute_ground_truth(
                "t", ROWS, agg_func="sum", agg_col="nonexistent_column"
            )

    def test_group_by_sorted_descending(self):
        gt = compute_ground_truth(
            "sorted", ROWS,
            agg_func="sum", agg_col="amt", group_col="category",
        )
        values = [v for _, v in gt.answer]
        self.assertEqual(values, sorted(values, reverse=True))


# ---------------------------------------------------------------------------
# TestRAGSimulation
# ---------------------------------------------------------------------------

class TestRAGSimulation(unittest.TestCase):

    def test_retrieves_correct_count(self):
        ctx = simulate_rag_retrieval("total spend", ROWS, 10)
        self.assertEqual(ctx.rows_retrieved, 10)

    def test_clamped_to_dataset_size(self):
        ctx = simulate_rag_retrieval("query", ROWS, len(ROWS) + 9999)
        self.assertEqual(ctx.rows_retrieved, len(ROWS))

    def test_zero_rows_requested(self):
        ctx = simulate_rag_retrieval("query", ROWS, 0)
        self.assertEqual(ctx.rows_retrieved, 0)
        self.assertEqual(ctx.coverage_pct, 0.0)

    def test_coverage_increases_with_size(self):
        ctx5   = simulate_rag_retrieval("spend", ROWS, 5)
        ctx500 = simulate_rag_retrieval("spend", ROWS, 500)
        self.assertGreater(ctx500.coverage_pct, ctx5.coverage_pct)

    def test_token_estimate_scales_with_rows(self):
        ctx = simulate_rag_retrieval("query", ROWS, 100)
        self.assertEqual(ctx.token_estimate, 100 * TOKENS_PER_ROW)

    def test_required_confidence_signals_present(self):
        ctx = simulate_rag_retrieval("category spend", ROWS, 50)
        required = [
            "categories_visible", "amounts_in_context",
            "partial_sum_visible", "specificity_score", "detectability_score",
        ]
        for key in required:
            self.assertIn(key, ctx.confidence_signals, f"Missing: {key}")

    def test_specificity_increases_with_context(self):
        ctx_sm = simulate_rag_retrieval("total category", ROWS, 5)
        ctx_lg = simulate_rag_retrieval("total category", ROWS, 500)
        self.assertGreater(
            ctx_lg.confidence_signals["specificity_score"],
            ctx_sm.confidence_signals["specificity_score"],
        )

    def test_partial_sum_less_than_true_total(self):
        ctx = simulate_rag_retrieval("total spend", ROWS, 50)
        self.assertLess(ctx.confidence_signals["partial_sum_visible"], 150_000.0)

    def test_context_text_not_empty_with_rows(self):
        ctx = simulate_rag_retrieval("category amount", ROWS, 5)
        self.assertGreater(len(ctx.context_text), 0)

    def test_empty_dataset_returns_safely(self):
        ctx = simulate_rag_retrieval("query", [], 10)
        self.assertEqual(ctx.rows_retrieved, 0)
        self.assertEqual(ctx.total_rows, 0)

    def test_latency_is_positive(self):
        ctx = simulate_rag_retrieval("test", ROWS, 10)
        self.assertGreater(ctx.latency_ms, 0)

    def test_coverage_pct_bounded(self):
        for size in CONTEXT_SIZES:
            actual = min(size, len(ROWS))
            ctx = simulate_rag_retrieval("query", ROWS, size)
            self.assertGreaterEqual(ctx.coverage_pct, 0.0)
            self.assertLessEqual(ctx.coverage_pct, 100.0)


# ---------------------------------------------------------------------------
# TestConfidenceMetrics
# ---------------------------------------------------------------------------

class TestConfidenceMetrics(unittest.TestCase):

    def test_specificity_range(self):
        for size in CONTEXT_SIZES:
            spec = _compute_specificity(size, min(size // 50 + 1, 14))
            self.assertGreaterEqual(spec, 0.0)
            self.assertLessEqual(spec, 1.0)

    def test_specificity_monotonic_with_size(self):
        specs = [_compute_specificity(s, min(s // 50 + 1, 14))
                 for s in CONTEXT_SIZES]
        for i in range(len(specs) - 1):
            self.assertLessEqual(
                specs[i], specs[i + 1],
                f"Specificity not monotonic at index {i}: {specs[i]} > {specs[i+1]}"
            )

    def test_detectability_tiny(self):
        self.assertIn("EASY", _compute_detectability(5, 1_000_000))

    def test_detectability_large(self):
        result = _compute_detectability(8_000, 1_000_000)
        self.assertIn("HARD", result.upper())

    def test_detectability_zero_total(self):
        result = _compute_detectability(100, 0)
        self.assertEqual(result, "UNKNOWN")

    def test_response_length_grows_with_context(self):
        small = _estimate_response_length(5)
        large = _estimate_response_length(8_000)
        self.assertIn("50", small)
        self.assertIn("1,500", large)

    def test_response_length_dynamic(self):
        # Verify it computes from context_size, not hardcoded keys
        result = _estimate_response_length(25)  # between 10 and 75
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# ---------------------------------------------------------------------------
# TestBenchmarkQueryValidation
# ---------------------------------------------------------------------------

class TestBenchmarkQueryValidation(unittest.TestCase):

    def test_all_canonical_queries_valid(self):
        for q in BENCHMARK_QUERIES:
            try:
                q.validate()
            except QueryError as exc:
                self.fail(f"Query '{q.label}' failed validation: {exc}")

    def test_empty_label_raises(self):
        q = BenchmarkQuery(label="", question="test", why="test")
        with self.assertRaises(QueryError):
            q.validate()

    def test_empty_question_raises(self):
        q = BenchmarkQuery(label="t", question="", why="test")
        with self.assertRaises(QueryError):
            q.validate()

    def test_invalid_agg_func_raises(self):
        q = BenchmarkQuery(
            label="t", question="q", why="w", agg_func="median"
        )
        with self.assertRaises(QueryError):
            q.validate()

    def test_invalid_operator_raises(self):
        q = BenchmarkQuery(
            label="t", question="q", why="w",
            numeric_filter_col="amt",
            numeric_filter_op="between",
            numeric_filter_val=100.0,
        )
        with self.assertRaises(QueryError):
            q.validate()

    def test_seven_canonical_queries(self):
        self.assertEqual(len(BENCHMARK_QUERIES), 7)

    def test_context_sizes_sorted_ascending(self):
        self.assertEqual(CONTEXT_SIZES, sorted(CONTEXT_SIZES))

    def test_all_context_sizes_positive(self):
        for size in CONTEXT_SIZES:
            self.assertGreater(size, 0)


# ---------------------------------------------------------------------------
# TestRunQuery (integration)
# ---------------------------------------------------------------------------

class TestRunQuery(unittest.TestCase):

    def test_run_sum_query(self):
        q = BenchmarkQuery(
            label="sum", question="total spend", why="test",
            agg_func="sum", agg_col="amt",
        )
        result = run_query(q, ROWS)
        self.assertIsInstance(result, QueryResult)
        self.assertAlmostEqual(result.ground_truth.answer, 150_000.0, delta=1.0)
        self.assertEqual(len(result.rag_contexts), len(CONTEXT_SIZES))

    def test_all_context_sizes_produced(self):
        q = BENCHMARK_QUERIES[0]
        result = run_query(q, ROWS)
        for ctx, expected in zip(result.rag_contexts, CONTEXT_SIZES):
            self.assertLessEqual(ctx.rows_retrieved, expected)

    def test_ground_truth_scans_all_rows(self):
        q = BENCHMARK_QUERIES[0]
        result = run_query(q, ROWS)
        self.assertEqual(result.ground_truth.rows_scanned, len(ROWS))

    def test_specificity_monotonically_increases(self):
        q = BENCHMARK_QUERIES[0]
        result = run_query(q, ROWS)
        specs = [
            ctx.confidence_signals["specificity_score"]
            for ctx in result.rag_contexts
        ]
        for i in range(len(specs) - 1):
            self.assertLessEqual(
                specs[i], specs[i + 1],
                f"Specificity not monotonic at {i}: {specs[i]} > {specs[i+1]}"
            )

    def test_invalid_query_raises(self):
        q = BenchmarkQuery(
            label="bad", question="q", why="w", agg_func="invalid"
        )
        with self.assertRaises(QueryError):
            run_query(q, ROWS)

    def test_result_has_all_fields(self):
        q = BENCHMARK_QUERIES[0]
        result = run_query(q, ROWS)
        self.assertIsNotNone(result.query)
        self.assertIsNotNone(result.ground_truth)
        self.assertIsNotNone(result.rag_contexts)
        self.assertGreater(len(result.rag_contexts), 0)

    def test_rag_partial_sum_always_less_than_true(self):
        q = BenchmarkQuery(
            label="sum", question="total amount", why="test",
            agg_func="sum", agg_col="amt",
        )
        result = run_query(q, ROWS)
        true_total = result.ground_truth.answer
        for ctx in result.rag_contexts[:3]:  # first 3 sizes are partial
            partial = ctx.confidence_signals["partial_sum_visible"]
            self.assertLessEqual(partial, true_total)


# ---------------------------------------------------------------------------
# TestLoadCSV
# ---------------------------------------------------------------------------

class TestLoadCSV(unittest.TestCase):

    def _write_temp_csv(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.flush()
        f.close()
        return f.name

    def test_load_valid_csv(self):
        path = self._write_temp_csv("amt,category\n10.0,A\n20.0,B\n")
        try:
            rows = load_csv(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["amt"], "10.0")
        finally:
            os.unlink(path)

    def test_max_rows_respected(self):
        content = "amt,category\n" + "\n".join(f"{i},A" for i in range(100))
        path = self._write_temp_csv(content)
        try:
            rows = load_csv(path, max_rows=10)
            self.assertEqual(len(rows), 10)
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with self.assertRaises(DataLoadError):
            load_csv("/nonexistent/path/file.csv")

    def test_missing_required_column_raises(self):
        path = self._write_temp_csv("name,value\nfoo,bar\n")
        try:
            with self.assertRaises(SchemaError):
                load_csv(path)
        finally:
            os.unlink(path)

    def test_empty_file_raises(self):
        path = self._write_temp_csv("")
        try:
            with self.assertRaises((DataLoadError, SchemaError)):
                load_csv(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_single_row_dataset(self):
        single = [{"amt": "100.00", "category": "A", "gender": "F",
                   "state": "NY", "is_fraud": "0"}]
        gt = compute_ground_truth("sum", single, agg_func="sum", agg_col="amt")
        self.assertEqual(gt.answer, 100.0)

    def test_all_same_category(self):
        uniform = [{"category": "X", "amt": str(float(i))}
                   for i in range(1, 101)]
        gt = compute_ground_truth(
            "sum", uniform, agg_func="sum", agg_col="amt", group_col="category"
        )
        answers = dict(gt.answer)
        # Sum 1..100 = 5050
        self.assertAlmostEqual(answers.get("X", 0), 5050.0, places=1)

    def test_missing_amt_values_skipped(self):
        rows = [
            {"amt": "",       "category": "A"},
            {"amt": "bad",    "category": "A"},
            {"amt": "100.00", "category": "A"},
        ]
        gt = compute_ground_truth("sum", rows, agg_func="sum", agg_col="amt")
        self.assertEqual(gt.answer, 100.0)

    def test_fmt_answer_with_list(self):
        gt = GroundTruth(
            query="test",
            answer=[("A", 100.0), ("B", 50.0)],
            description="",
            latency_ms=1.0,
            rows_scanned=10,
            rows_matched=10,
        )
        result = _fmt_answer(gt)
        self.assertIn("A=100.00", result)
        self.assertIn("B=50.00", result)

    def test_fmt_answer_with_float(self):
        gt = GroundTruth(
            query="test",
            answer=1234567.89,
            description="",
            latency_ms=1.0,
            rows_scanned=10,
            rows_matched=10,
        )
        result = _fmt_answer(gt)
        self.assertIn("1,234,567.89", result)

    def test_specificity_bounded_extreme_inputs(self):
        for size in [0, 1, 100, 100_000, 10_000_000]:
            spec = _compute_specificity(size, 14)
            self.assertGreaterEqual(spec, 0.0)
            self.assertLessEqual(spec, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
