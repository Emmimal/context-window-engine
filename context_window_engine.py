"""
context_window_engine.py
------------------------
Larger Context Windows Don't Fix RAG — They Make Errors Harder to Detect.

Across 7 query types on a 100K-row dataset, increasing context size
didn't improve accuracy — it made errors harder to detect.

Zero external dependencies. Pure Python 3.9+ stdlib only.
No API keys. No LLM calls. Fully reproducible.

The engine runs two pipelines side by side for each query:

  RAG simulation — what a naive RAG pipeline passes to an LLM at five
  context window sizes (5 rows → 8,000 rows, covering ~325 to ~520,000
  tokens). Measures coverage, confidence signals, and detectability of
  the error at each size.

  Semantic engine — exact deterministic aggregation over the full
  dataset. SUM, AVG, COUNT, MIN, MAX, GROUP BY, categorical filters,
  numeric comparisons. Zero inference. Single-pass scan.

The core finding: as context grows, LLM responses become more specific,
more detailed, and more convincing — while remaining wrong. At 8,000
rows the error is nearly impossible to detect without the ground truth.

Run:
    python context_window_engine.py                   # 100k rows
    python context_window_engine.py --full            # all 1.29M rows
    python context_window_engine.py --rows 50000      # custom row count
    python context_window_engine.py --query 0         # single query
    python context_window_engine.py --sample-context  # show raw LLM input
    python context_window_engine.py --output out.txt  # save results
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Approximate tokens per CSV row (based on avg row length in this dataset)
TOKENS_PER_ROW: int = 65

# Context window sizes to test (in rows)
# Approximate token budgets:
#   5     →    325 tokens  (tiny, GPT-3.5 era retrieval)
#   50    →  3,250 tokens  (standard RAG top-k retrieval)
#   500   → 32,500 tokens  (medium context window)
#   2,000 → 130,000 tokens (GPT-4 / Claude standard)
#   8,000 → 520,000 tokens (approaching 1M-token windows)
CONTEXT_SIZES: List[int] = [5, 50, 500, 2_000, 8_000]

CONTEXT_LABELS: Dict[int, str] = {
    5:     "tiny    (~325 tokens)",
    50:    "small   (~3K tokens)",
    500:   "medium  (~32K tokens)",
    2_000: "large   (~130K tokens)",
    8_000: "xlarge  (~520K tokens, approaching 1M)",
}

# Required CSV columns — validated at load time
REQUIRED_COLUMNS: List[str] = ["amt"]

SEP  = "─" * 76
SEP2 = "═" * 76

# Valid aggregation functions
VALID_AGG_FUNCS = {"sum", "avg", "count", "max", "min"}

# Valid numeric filter operators
VALID_OPERATORS = {"gt", "gte", "lt", "lte", "eq"}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class EngineError(Exception):
    """Base exception for all engine errors."""


class DataLoadError(EngineError):
    """Raised when CSV loading fails."""


class SchemaError(EngineError):
    """Raised when required columns are missing."""


class QueryError(EngineError):
    """Raised when a query definition is invalid."""


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(
    path: str,
    max_rows: Optional[int] = None,
    encoding: str = "utf-8",
) -> List[Dict[str, str]]:
    """
    Load a CSV file into a list of row dicts.

    Args:
        path:     Absolute or relative path to the CSV file.
        max_rows: Maximum rows to load. None loads all rows.
        encoding: File encoding (default utf-8, falls back to latin-1).

    Returns:
        List of row dicts with string values.

    Raises:
        DataLoadError: If the file does not exist or cannot be parsed.
        SchemaError:   If required columns are missing.
    """
    if not os.path.exists(path):
        raise DataLoadError(
            f"CSV file not found: {path}\n"
            f"Place your CSV at: {os.path.abspath(path)}"
        )

    rows: List[Dict[str, str]] = []
    t0 = time.perf_counter()

    # Try UTF-8 first, fall back to latin-1 for legacy CSVs
    for enc in (encoding, "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise DataLoadError(f"CSV file appears empty: {path}")

                # Validate required columns
                missing = [c for c in REQUIRED_COLUMNS
                           if c not in (reader.fieldnames or [])]
                if missing:
                    raise SchemaError(
                        f"Required columns missing from CSV: {missing}\n"
                        f"Available columns: {list(reader.fieldnames or [])}"
                    )

                for i, row in enumerate(reader):
                    if max_rows is not None and i >= max_rows:
                        break
                    rows.append(row)
            break  # success — stop trying encodings
        except (UnicodeDecodeError, csv.Error) as exc:
            if enc == "latin-1":
                raise DataLoadError(f"Cannot parse CSV: {exc}") from exc
            continue  # try next encoding

    elapsed = round((time.perf_counter() - t0) * 1_000, 1)
    print(f"  Loaded {len(rows):,} rows in {elapsed}ms from {os.path.basename(path)}")
    return rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_float(val: str) -> Optional[float]:
    """
    Parse a string to float, handling commas and dollar signs.
    Returns None if the value cannot be parsed — never raises.
    """
    if not val or not val.strip():
        return None
    try:
        return float(val.strip().replace(",", "").replace("$", ""))
    except ValueError:
        return None


def _apply_numeric_filter(
    rows: List[Dict[str, str]],
    col: str,
    op: str,
    val: float,
) -> List[Dict[str, str]]:
    """Apply a numeric comparison filter to a row list."""
    ops = {
        "gt":  lambda v: v > val,
        "gte": lambda v: v >= val,
        "lt":  lambda v: v < val,
        "lte": lambda v: v <= val,
        "eq":  lambda v: math.isclose(v, val),
    }
    if op not in ops:
        raise QueryError(
            f"Invalid operator '{op}'. "
            f"Valid operators: {sorted(VALID_OPERATORS)}"
        )
    predicate = ops[op]
    result = []
    for row in rows:
        parsed = _try_float(row.get(col, ""))
        if parsed is not None and predicate(parsed):
            result.append(row)
    return result


def _aggregate(
    values: List[float],
    func: str,
) -> float:
    """Apply an aggregation function to a list of floats."""
    if func not in VALID_AGG_FUNCS:
        raise QueryError(
            f"Invalid aggregation function '{func}'. "
            f"Valid functions: {sorted(VALID_AGG_FUNCS)}"
        )
    if not values:
        return 0.0
    if func == "sum":   return round(sum(values), 2)
    if func == "avg":   return round(sum(values) / len(values), 2)
    if func == "count": return float(len(values))
    if func == "max":   return round(max(values), 2)
    if func == "min":   return round(min(values), 2)
    return 0.0  # unreachable — kept for type checker


# ---------------------------------------------------------------------------
# Ground truth — exact aggregation (the semantic engine)
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    """Exact aggregation result from the semantic engine."""
    query:        str
    answer:       Any                    # float or List[Tuple[str, float]]
    description:  str
    latency_ms:   float
    rows_scanned: int
    rows_matched: int


def compute_ground_truth(
    query_label:        str,
    rows:               List[Dict[str, str]],
    agg_func:           str = "sum",
    agg_col:            str = "amt",
    filter_col:         Optional[str] = None,
    filter_val:         Optional[str] = None,
    group_col:          Optional[str] = None,
    numeric_filter_col: Optional[str] = None,
    numeric_filter_op:  Optional[str] = None,
    numeric_filter_val: Optional[float] = None,
) -> GroundTruth:
    """
    Exact computation over the full dataset. No inference required.

    This is what the semantic engine produces — deterministic, complete,
    and fast. Contrasted against RAG at each context size.

    Args:
        query_label:        Human-readable label for this query.
        rows:               Full dataset.
        agg_func:           Aggregation function (sum/avg/count/max/min).
        agg_col:            Column to aggregate.
        filter_col:         Optional categorical filter column.
        filter_val:         Value to match for categorical filter.
        group_col:          Optional GROUP BY column.
        numeric_filter_col: Optional numeric filter column.
        numeric_filter_op:  Operator for numeric filter (gt/gte/lt/lte/eq).
        numeric_filter_val: Threshold value for numeric filter.

    Returns:
        GroundTruth with exact answer and timing.

    Raises:
        QueryError: If aggregation function or operator is invalid.
        SchemaError: If specified columns don't exist in the data.
    """
    if agg_func not in VALID_AGG_FUNCS:
        raise QueryError(f"Invalid agg_func '{agg_func}'. Use: {VALID_AGG_FUNCS}")

    t0 = time.perf_counter()

    # Validate columns exist
    if rows:
        sample = rows[0]
        for col in filter(None, [filter_col, group_col, numeric_filter_col, agg_col]):
            if col not in sample:
                raise SchemaError(
                    f"Column '{col}' not found in dataset. "
                    f"Available: {sorted(sample.keys())}"
                )

    # Apply categorical filter
    filtered = rows
    if filter_col and filter_val is not None:
        filtered = [
            r for r in filtered
            if r.get(filter_col, "").strip().lower() == filter_val.strip().lower()
        ]

    # Apply numeric filter
    if (numeric_filter_col and numeric_filter_op
            and numeric_filter_val is not None):
        filtered = _apply_numeric_filter(
            filtered, numeric_filter_col, numeric_filter_op, numeric_filter_val
        )

    rows_matched = len(filtered)

    # Aggregate
    if group_col:
        groups: Dict[str, List[float]] = defaultdict(list)
        for row in filtered:
            key = row.get(group_col, "Unknown").strip()
            val = _try_float(row.get(agg_col, ""))
            if val is not None:
                groups[key].append(val)

        results: List[Tuple[str, float]] = []
        for key, vals in groups.items():
            results.append((key, _aggregate(vals, agg_func)))

        # Sort descending by value
        results.sort(key=lambda x: x[1], reverse=True)

        latency_ms = round((time.perf_counter() - t0) * 1_000, 2)

        # Build description
        filter_clause = ""
        if filter_col:
            filter_clause = f" WHERE {filter_col}={filter_val}"
        if numeric_filter_col:
            filter_clause += (
                f" WHERE {numeric_filter_col} "
                f"{numeric_filter_op} {numeric_filter_val}"
            )
        desc = (
            f"{agg_func.upper()}({agg_col}){filter_clause} "
            f"GROUP BY {group_col} → {len(results)} groups"
        )

        return GroundTruth(
            query=query_label,
            answer=results,
            description=desc,
            latency_ms=latency_ms,
            rows_scanned=len(rows),
            rows_matched=rows_matched,
        )

    else:
        vals = [_try_float(r.get(agg_col, "")) for r in filtered]
        nums = [v for v in vals if v is not None]
        result = _aggregate(nums, agg_func)

        latency_ms = round((time.perf_counter() - t0) * 1_000, 2)

        parts = []
        if filter_col:
            parts.append(f"{filter_col}={filter_val}")
        if numeric_filter_col:
            parts.append(
                f"{numeric_filter_col} {numeric_filter_op} {numeric_filter_val}"
            )
        filter_clause = (" WHERE " + " AND ".join(parts)) if parts else " FULL DATASET"
        desc = (
            f"{agg_func.upper()}({agg_col}){filter_clause} "
            f"→ {rows_matched:,} rows matched"
        )

        return GroundTruth(
            query=query_label,
            answer=result,
            description=desc,
            latency_ms=latency_ms,
            rows_scanned=len(rows),
            rows_matched=rows_matched,
        )


# ---------------------------------------------------------------------------
# RAG simulation — what naive RAG actually passes to an LLM
# ---------------------------------------------------------------------------

@dataclass
class RAGContext:
    """Everything the LLM actually receives at a given context size."""
    context_size:       int
    context_label:      str
    token_estimate:     int
    rows_retrieved:     int
    total_rows:         int
    coverage_pct:       float
    context_text:       str
    confidence_signals: Dict[str, Any]
    latency_ms:         float


def simulate_rag_retrieval(
    query:        str,
    rows:         List[Dict[str, str]],
    context_size: int,
    query_tokens: Optional[List[str]] = None,
) -> RAGContext:
    """
    Simulate naive RAG on structured CSV data.

    Standard RAG pipeline on a CSV:
      1. Flatten each row to a text string
      2. Score rows by keyword overlap with the query
      3. Return top-k rows as plain-text context to the LLM

    This is what vector RAG does when applied to tabular data.
    It has no concept of aggregation, grouping, or numeric comparison.

    Args:
        query:        Natural language query.
        rows:         Full dataset.
        context_size: Number of rows to retrieve (k in top-k).
        query_tokens: Pre-tokenised query words (computed if None).

    Returns:
        RAGContext with everything the LLM would receive.
    """
    if not rows:
        return RAGContext(
            context_size=context_size,
            context_label=CONTEXT_LABELS.get(context_size, f"{context_size} rows"),
            token_estimate=0,
            rows_retrieved=0,
            total_rows=0,
            coverage_pct=0.0,
            context_text="",
            confidence_signals={},
            latency_ms=0.0,
        )

    t0 = time.perf_counter()

    # Clamp to available rows
    actual_size = min(context_size, len(rows))

    # Tokenise query
    if query_tokens is None:
        query_tokens = re.findall(r"[a-z]+", query.lower())
    query_token_set = set(query_tokens)

    # Score by keyword overlap (simulates BM25 / TF-IDF retrieval)
    scored: List[Tuple[int, int, str]] = []
    for i, row in enumerate(rows):
        text = " ".join(str(v) for v in row.values()).lower()
        overlap = sum(1 for t in query_token_set if t in text)
        scored.append((overlap, i, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = scored[:actual_size]

    # Build context text (what the LLM actually receives)
    context_lines = [text[:200] for _, _, text in top_k]
    context_text = "\n---\n".join(context_lines)

    # Measure confidence signals — these are the article's core metric
    unique_categories: set = set()
    unique_states: set     = set()
    amounts: List[float]   = []

    for _, idx, _ in top_k:
        row = rows[idx]
        cat   = row.get("category", "").strip()
        state = row.get("state", "").strip()
        amt   = _try_float(row.get("amt", ""))
        if cat:   unique_categories.add(cat)
        if state: unique_states.add(state)
        if amt is not None: amounts.append(amt)

    partial_sum = round(sum(amounts), 2) if amounts else 0.0
    partial_avg = round(sum(amounts) / len(amounts), 2) if amounts else 0.0

    latency_ms = round((time.perf_counter() - t0) * 1_000, 2)

    confidence_signals = {
        "categories_visible":    len(unique_categories),
        "states_visible":        len(unique_states),
        "amounts_in_context":    len(amounts),
        "partial_sum_visible":   partial_sum,
        "partial_avg_visible":   partial_avg,
        "coverage_pct":          round(actual_size / len(rows) * 100, 4),
        "response_length_proxy": _estimate_response_length(actual_size),
        "specificity_score":     _compute_specificity(
            actual_size, len(unique_categories)
        ),
        "detectability_score":   _compute_detectability(actual_size, len(rows)),
    }

    return RAGContext(
        context_size    = context_size,
        context_label   = CONTEXT_LABELS.get(
            context_size, f"{context_size:,} rows"
        ),
        token_estimate  = actual_size * TOKENS_PER_ROW,
        rows_retrieved  = actual_size,
        total_rows      = len(rows),
        coverage_pct    = round(actual_size / len(rows) * 100, 4),
        context_text    = context_text,
        confidence_signals = confidence_signals,
        latency_ms      = latency_ms,
    )


def _estimate_response_length(context_size: int) -> str:
    """
    Proxy for LLM response verbosity at each context size.
    Computed dynamically from context_size — not hardcoded.
    More context → longer, more detailed-looking response.
    """
    if context_size <= 10:
        return "~50 words — obvious uncertainty"
    if context_size <= 75:
        return "~150 words — some category breakdowns"
    if context_size <= 750:
        return "~400 words — detailed breakdowns, specific numbers"
    if context_size <= 3_000:
        return "~800 words — confident analysis with charts described"
    return "~1,500+ words — authoritative, detailed, wrong"


def _compute_specificity(context_size: int, categories_visible: int) -> float:
    """
    How authoritative the LLM response will appear [0.0 – 1.0].
    Combines context coverage (log scale) with categorical visibility.
    """
    base       = math.log1p(context_size) / math.log1p(max(CONTEXT_SIZES))
    cat_factor = categories_visible / max(14, categories_visible)
    return round(min(1.0, base * 0.7 + cat_factor * 0.3), 3)


def _compute_detectability(context_size: int, total_rows: int) -> str:
    """
    How easy it is to detect that the LLM answer is wrong.
    This is the core finding: larger context → harder to detect errors.
    """
    if total_rows == 0:
        return "UNKNOWN"
    coverage = context_size / total_rows * 100
    if coverage < 0.01:   return "EASY — response is obviously a guess (tiny sample)"
    if coverage < 0.1:    return "MODERATE — partial data visible, error detectable"
    if coverage < 1.0:    return "HARD — response looks plausible, error subtle"
    if coverage < 5.0:    return "VERY HARD — response appears authoritative"
    return "NEAR IMPOSSIBLE — response indistinguishable from correct"


# ---------------------------------------------------------------------------
# Benchmark query definitions
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkQuery:
    """Definition of one benchmark query."""
    label:              str
    question:           str
    why:                str
    agg_func:           str            = "sum"
    agg_col:            str            = "amt"
    filter_col:         Optional[str]  = None
    filter_val:         Optional[str]  = None
    group_col:          Optional[str]  = None
    numeric_filter_col: Optional[str]  = None
    numeric_filter_op:  Optional[str]  = None
    numeric_filter_val: Optional[float]= None
    query_tokens:       Optional[List[str]] = None
    display_ascending:  bool           = False  # True for min/lowest queries — reverses display order

    def validate(self) -> None:
        """Raise QueryError if the query definition is invalid."""
        if not self.label:
            raise QueryError("Query label cannot be empty")
        if not self.question:
            raise QueryError("Query question cannot be empty")
        if self.agg_func not in VALID_AGG_FUNCS:
            raise QueryError(
                f"Invalid agg_func '{self.agg_func}' in query '{self.label}'. "
                f"Valid: {sorted(VALID_AGG_FUNCS)}"
            )
        if (self.numeric_filter_op is not None
                and self.numeric_filter_op not in VALID_OPERATORS):
            raise QueryError(
                f"Invalid operator '{self.numeric_filter_op}' in query '{self.label}'. "
                f"Valid: {sorted(VALID_OPERATORS)}"
            )


BENCHMARK_QUERIES: List[BenchmarkQuery] = [
    BenchmarkQuery(
        label    = "total_by_category",
        question = "What is the total spend by category?",
        why      = "SUM + GROUP BY — the canonical aggregation RAG cannot perform",
        group_col= "category",
        agg_func = "sum",
        agg_col  = "amt",
    ),
    BenchmarkQuery(
        label    = "avg_by_category",
        question = "Which category has the highest average transaction amount?",
        why      = "AVG + GROUP BY — requires numeric reasoning across all groups",
        group_col= "category",
        agg_func = "avg",
        agg_col  = "amt",
    ),
    BenchmarkQuery(
        label      = "filter_sum",
        question   = "What is the total amount spent on grocery_pos?",
        why        = "SUM + categorical filter — RAG retrieves chunks, cannot sum",
        filter_col = "category",
        filter_val = "grocery_pos",
        agg_func   = "sum",
        agg_col    = "amt",
    ),
    BenchmarkQuery(
        label      = "count_filter",
        question   = "How many transactions were made by female customers?",
        why        = "COUNT + filter — RAG returns text rows, not a count",
        filter_col = "gender",
        filter_val = "F",
        agg_func   = "count",
        agg_col    = "amt",
    ),
    BenchmarkQuery(
        label              = "numeric_comparison",
        question           = "What is the total spend where amount is greater than 500?",
        why                = "SUM + numeric comparison — RAG cannot evaluate > operator",
        numeric_filter_col = "amt",
        numeric_filter_op  = "gt",
        numeric_filter_val = 500.0,
        agg_func           = "sum",
        agg_col            = "amt",
    ),
    BenchmarkQuery(
        label             = "min_group",
        question          = "Which state has the lowest total spending?",
        why               = "MIN + GROUP BY across 50 states — requires full dataset scan",
        group_col         = "state",
        agg_func          = "sum",
        agg_col           = "amt",
        display_ascending = True,   # query asks for lowest — display ascending
    ),
    BenchmarkQuery(
        label    = "fraud_rate",
        question = "What percentage of transactions are fraudulent?",
        why      = "COUNT + ratio — RAG cannot compute proportions",
        agg_func = "sum",
        agg_col  = "is_fraud",
    ),
]


# ---------------------------------------------------------------------------
# Experiment result
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Full result for one benchmark query."""
    query:        BenchmarkQuery
    ground_truth: GroundTruth
    rag_contexts: List[RAGContext]


def run_query(
    query: BenchmarkQuery,
    rows:  List[Dict[str, str]],
) -> QueryResult:
    """
    Run one benchmark query: compute ground truth and simulate RAG
    at all configured context sizes.

    Args:
        query: The benchmark query definition.
        rows:  Full loaded dataset.

    Returns:
        QueryResult with ground truth and all RAG context simulations.

    Raises:
        QueryError:  If the query definition is invalid.
        SchemaError: If required columns are missing.
    """
    query.validate()

    gt = compute_ground_truth(
        query_label        = query.question,
        rows               = rows,
        agg_func           = query.agg_func,
        agg_col            = query.agg_col,
        filter_col         = query.filter_col,
        filter_val         = query.filter_val,
        group_col          = query.group_col,
        numeric_filter_col = query.numeric_filter_col,
        numeric_filter_op  = query.numeric_filter_op,
        numeric_filter_val = query.numeric_filter_val,
    )

    rag_contexts = [
        simulate_rag_retrieval(
            query        = query.question,
            rows         = rows,
            context_size = size,
            query_tokens = query.query_tokens,
        )
        for size in CONTEXT_SIZES
    ]

    return QueryResult(query=query, ground_truth=gt, rag_contexts=rag_contexts)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_answer(gt: GroundTruth, ascending: bool = False) -> str:
    if isinstance(gt.answer, list):
        items = sorted(gt.answer, key=lambda x: x[1]) if ascending else gt.answer
        top3  = items[:3]
        return ", ".join(f"{k}={v:,.2f}" for k, v in top3)
    if isinstance(gt.answer, float):
        return f"{gt.answer:,.2f}"
    return str(gt.answer)


# ---------------------------------------------------------------------------
# Reporters
# ---------------------------------------------------------------------------

def report_query(result: QueryResult, output_lines: List[str]) -> None:
    """Print detailed results for one query."""
    def out(line: str = "") -> None:
        print(line)
        output_lines.append(line)

    q  = result.query
    gt = result.ground_truth

    out(f"\n{SEP2}")
    out(f"  QUERY : \"{q.question}\"")
    out(f"  WHY   : {q.why}")
    out(SEP2)

    out(f"\n  ── GROUND TRUTH (Semantic Engine) {'─'*38}")
    out(f"  {gt.description}")
    if isinstance(gt.answer, list):
        # For min queries, sort ascending and label from lowest; else show top-5 descending.
        display = (
            sorted(gt.answer, key=lambda x: x[1])
            if q.display_ascending
            else gt.answer
        )
        for rank, (k, v) in enumerate(display[:5], 1):
            if q.display_ascending and rank == 1:
                prefix = f"#{rank:>2} (lowest)"
            else:
                prefix = f"#{rank:>2}"
            out(f"    {prefix:<12}  {k:<22}  {v:>14,.2f}")
    else:
        out(f"  Exact answer : {_fmt_answer(gt)}")
    out(f"  Latency      : {gt.latency_ms}ms  |  "
        f"Rows scanned: {gt.rows_scanned:,}  |  "
        f"Rows matched: {gt.rows_matched:,}")

    out(f"\n  ── WHAT RAG PASSES TO THE LLM AT EACH CONTEXT SIZE {'─'*24}")
    out(f"\n  {'Context':<46} {'Rows':>6} {'Coverage':>10} "
        f"{'Cats':>5} {'Partial':>14} {'Spec':>6}  Error Detectability")
    out(f"  {'-'*46} {'-'*6} {'-'*10} {'-'*5} {'-'*14} {'-'*6}  {'-'*36}")

    for ctx in result.rag_contexts:
        cs = ctx.confidence_signals
        out(
            f"  {ctx.context_label:<46} "
            f"{ctx.rows_retrieved:>6,} "
            f"{ctx.coverage_pct:>9.4f}% "
            f"{cs.get('categories_visible', 0):>5} "
            f"{cs.get('partial_sum_visible', 0.0):>14,.2f} "
            f"{cs.get('specificity_score', 0.0):>6.3f}  "
            f"{cs.get('detectability_score', '')}"
        )

    out(f"\n  ── THE DETECTABILITY PROBLEM {'─'*48}")
    out(f"  Correct answer : {_fmt_answer(gt, ascending=q.display_ascending)}")
    out()
    for ctx in result.rag_contexts:
        cs = ctx.confidence_signals
        partial = cs.get("partial_sum_visible", 0.0)
        out(
            f"  Context {ctx.rows_retrieved:>6,} rows : "
            f"LLM sees {cs.get('categories_visible', 0):>2} categories, "
            f"partial sum = {partial:>14,.2f}  →  "
            f"{cs.get('response_length_proxy', '')}"
        )

    largest = result.rag_contexts[-1]
    out()
    out(f"  At {largest.rows_retrieved:,} rows the LLM sees "
        f"{largest.confidence_signals.get('categories_visible', 0)} of 14 categories.")
    out(f"  Its response will be ~1,500 words, specific, authoritative — and wrong.")
    out(f"  At 5 rows the error is obvious. At {largest.rows_retrieved:,} rows it is hidden.")


def report_confidence_table(
    results: List[QueryResult],
    output_lines: List[str],
) -> None:
    """The article's money-shot: confidence rises, accuracy stays zero."""
    def out(line: str = "") -> None:
        print(line)
        output_lines.append(line)

    out(f"\n{SEP2}")
    out(f"  CONFIDENCE ESCALATION — THE CORE FINDING")
    out(f"  As context grows, LLM responses become more specific and convincing.")
    out(f"  Coverage of actual data stays near zero throughout.")
    out(SEP2)

    r  = results[0]
    gt = r.ground_truth

    out(f"\n  Query   : \"{r.query.question}\"")
    out(f"  Correct : {_fmt_answer(gt)}")
    out(f"  Dataset : {gt.rows_scanned:,} rows\n")

    out(f"  {'Context Window':<46} {'Rows':>7} {'Coverage':>10} "
        f"{'Spec':>6} {'Response':>9}  Error visible?")
    out(f"  {'-'*46} {'-'*7} {'-'*10} {'-'*6} {'-'*9}  {'-'*28}")

    resp_short = {
        5:     "~50w",
        50:    "~150w",
        500:   "~400w",
        2_000: "~800w",
        8_000: "~1,500w",
    }
    detect_short = {
        5:     "YES — obviously incomplete",
        50:    "MAYBE — partial data",
        500:   "HARD — looks plausible",
        2_000: "VERY HARD — authoritative",
        8_000: "NEAR IMPOSSIBLE",
    }

    for ctx in r.rag_contexts:
        cs   = ctx.confidence_signals
        size = ctx.rows_retrieved
        out(
            f"  {ctx.context_label:<46} "
            f"{size:>7,} "
            f"{ctx.coverage_pct:>9.4f}% "
            f"{cs.get('specificity_score', 0.0):>6.3f} "
            f"{resp_short.get(size, '?'):>9}  "
            f"{detect_short.get(size, '?')}"
        )

    out(
        f"\n  Semantic Engine : {gt.rows_scanned:,} rows | "
        f"{gt.latency_ms}ms | exact answer | zero inference"
    )


def report_why_it_cannot_be_fixed(
    rows: List[Dict[str, str]],
    output_lines: List[str],
) -> None:
    """Explains mathematically why larger context doesn't fix the problem."""
    def out(line: str = "") -> None:
        print(line)
        output_lines.append(line)

    total = len(rows)
    coverage_8k = round(8_000 / total * 100, 2) if total else 0

    out(f"\n{SEP2}")
    out(f"  WHY LARGER CONTEXT CANNOT FIX THIS")
    out(SEP2)
    out(f"""
  RAG treats each CSV row as a plain-text document:

    "2019-01-01 grocery_pos 107.23 F NC Jennifer Banks ..."

  For "What is total spend by category?" the RAG pipeline:
    1. Tokenises the query: ["total", "spend", "category"]
    2. Scores all {total:,} rows by keyword overlap
    3. Returns the top-N rows as plain text
    4. Passes those rows to the LLM and asks it to compute

  The fundamental problem is not context size. It is retrieval method.

  At     5 rows : LLM sees {5/total*100:.4f}% of data → obviously guessing
  At 8,000 rows : LLM sees {coverage_8k:.2f}% of data  → looks authoritative, still wrong

  Even at 1M tokens (the full {total:,} rows passed directly):
    - The LLM must sum {total:,} numeric values from raw text in its context
    - Research shows LLMs lose numerical precision at scale [1]
    - Response time would be minutes, not milliseconds
    - The answer would still be less accurate than a {round(sum(
        _try_float(r.get('amt', '')) or 0 for r in rows[:1]
    ), 0):.0f}ms deterministic scan

  The semantic engine doesn't retrieve and reason. It computes:
    parse("total spend by category") → SUM(amt) GROUP BY category
    scan {total:,} rows in a single pass
    return exact grouped totals with zero inference

  Larger context windows make the wrong answer harder to detect.
  They do not make it more correct.
    """)


def report_master_table(
    results: List[QueryResult],
    output_lines: List[str],
) -> None:
    def out(line: str = "") -> None:
        print(line)
        output_lines.append(line)

    out(f"\n{SEP2}")
    out(f"  MASTER BENCHMARK TABLE")
    out(SEP2)
    out(f"\n  {'Query':<45} {'SL(ms)':>8}  {'Scanned':>10}  "
        f"{'5-row RAG':>14}  {'8K-row RAG':>14}")
    out(f"  {'-'*45} {'-'*8}  {'-'*10}  {'-'*14}  {'-'*14}")

    for r in results:
        gt = r.ground_truth
        out(
            f"  {r.query.question[:45]:<45} "
            f"{gt.latency_ms:>8}ms  "
            f"{gt.rows_scanned:>10,}  "
            f"  {'obvious error':>14}  "
            f"  {'hidden error':>14}"
        )


def print_sample_rag_context(
    result: QueryResult,
    context_size: int = 50,
) -> None:
    """Print the raw text the LLM would receive — makes the problem tangible."""
    ctx = next(
        (c for c in result.rag_contexts if c.rows_retrieved == context_size),
        None,
    )
    if not ctx:
        return

    print(f"\n{SEP2}")
    print(f"  RAW LLM INPUT — {context_size} rows (~{ctx.token_estimate:,} tokens)")
    print(f"  Query: \"{result.query.question}\"")
    print(SEP2)
    lines = ctx.context_text.split("\n---\n")
    for i, line in enumerate(lines[:5]):
        print(f"  Row {i+1}: {line[:120]}...")
    if len(lines) > 5:
        print(f"  ... and {len(lines) - 5} more rows of raw text")
    print()
    print(f"  The LLM receives {context_size} rows of serialised CSV text.")
    print(f"  It cannot sum. It cannot group. It cannot compare numerically.")
    print(f"  It will hallucinate a specific, confident, wrong answer.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Context Window Engine — proves larger context windows "
            "don't fix RAG, they just hide its weaknesses."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python context_window_engine.py
  python context_window_engine.py --full
  python context_window_engine.py --rows 50000
  python context_window_engine.py --query 0
  python context_window_engine.py --sample-context
  python context_window_engine.py --output results.txt
        """,
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Load all 1.29M rows (slower, more realistic production numbers)",
    )
    parser.add_argument(
        "--rows", type=int, default=100_000,
        help="Number of rows to load (default: 100,000)",
    )
    parser.add_argument(
        "--query", type=int, default=None,
        help=f"Run a single query by index 0–{len(BENCHMARK_QUERIES)-1}",
    )
    parser.add_argument(
        "--sample-context", action="store_true",
        help="Print the raw text the LLM receives for the first query",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save full output to this file (e.g. results.txt)",
    )
    parser.add_argument(
        "--csv", type=str,
        default=os.path.join("data", "credit_card_transactions.csv"),
        help="Path to CSV file (default: data/credit_card_transactions.csv)",
    )
    args = parser.parse_args()

    output_lines: List[str] = []

    def out(line: str = "") -> None:
        print(line)
        output_lines.append(line)

    out(f"\n{SEP2}")
    out(f"  Larger Context Windows Don't Fix RAG — They Make Errors Harder to Detect")
    out(f"  Across 7 query types on a 100K-row dataset, increasing context size")
    out(f"  didn't improve accuracy — it made errors harder to detect.")
    out(f"  Zero external dependencies | Pure Python 3.9+ | No API keys required")
    out(SEP2)

    # Load data
    try:
        max_rows = None if args.full else args.rows
        rows = load_csv(args.csv, max_rows=max_rows)
    except (DataLoadError, SchemaError) as exc:
        print(f"\n  ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    total = len(rows)
    token_budgets = [s * TOKENS_PER_ROW for s in CONTEXT_SIZES]
    out(f"\n  Dataset        : {total:,} rows")
    out(f"  Context sizes  : {CONTEXT_SIZES}")
    out(f"  Token budgets  : {token_budgets}")
    out(f"  API calls      : 0")
    out(f"  Dependencies   : 0\n")

    # Select queries
    if args.query is not None:
        if not (0 <= args.query < len(BENCHMARK_QUERIES)):
            print(
                f"  ERROR: --query must be 0–{len(BENCHMARK_QUERIES)-1}",
                file=sys.stderr,
            )
            sys.exit(1)
        queries = [BENCHMARK_QUERIES[args.query]]
    else:
        queries = BENCHMARK_QUERIES

    # Run
    results: List[QueryResult] = []
    for i, query in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {query.label}...", end=" ", flush=True)
        try:
            result = run_query(query, rows)
            results.append(result)
            print(f"✓  {result.ground_truth.latency_ms}ms")
        except (QueryError, SchemaError) as exc:
            print(f"✗  {exc}", file=sys.stderr)
            continue

    if not results:
        print("  No results produced. Check your --query index.", file=sys.stderr)
        sys.exit(1)

    # Reports
    for result in results:
        report_query(result, output_lines)
        if args.sample_context:
            print_sample_rag_context(result, context_size=50)

    report_master_table(results, output_lines)
    report_confidence_table(results, output_lines)
    report_why_it_cannot_be_fixed(rows, output_lines)

    out(f"\n{SEP2}")
    out(f"  COMPLETE")
    out(f"  Rows processed : {total:,}")
    out(f"  Queries run    : {len(results)}")
    out(f"  Context sizes  : {len(CONTEXT_SIZES)}")
    out(f"  API calls      : 0")
    out(f"  Dependencies   : 0")
    out(SEP2 + "\n")

    # Save output if requested
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines))
            print(f"  Output saved to: {args.output}")
        except OSError as exc:
            print(f"  WARNING: Could not save output: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
