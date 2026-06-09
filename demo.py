"""
demo.py
-------
Single-command demonstration of the Context Window Engine + Query Router.

Part 1 — The Problem:
  Shows Error Observability Collapse across two benchmark queries.
  RAG answers become more authoritative as context grows — while
  staying structurally wrong.

Part 2 — The Solution:
  The QueryRouter classifies the same two queries, dispatches them
  to the SemanticEngine, and returns the exact correct answer.

Run:
    python demo.py
"""

import os
import re
import sys
sys.path.insert(0, os.path.dirname(__file__))

from context_window_engine import (
    BENCHMARK_QUERIES,
    load_csv,
    run_query,
    _fmt_answer,
)
from query_router import QueryRouter, Route

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "credit_card_transactions.csv")
SEP  = "─" * 72
SEP2 = "═" * 72


def section_problem(rows) -> None:
    """Part 1: prove the problem — RAG error grows invisible with context."""

    demo_queries = [
        BENCHMARK_QUERIES[0],  # total spend by category (SUM + GROUP BY)
        BENCHMARK_QUERIES[4],  # total spend > $500 (SUM + numeric filter)
    ]

    for query in demo_queries:
        print(f"\n{SEP}")
        print(f"  Query : \"{query.question}\"")
        print(f"  Why   : {query.why}")
        print(SEP)

        result = run_query(query, rows)
        gt     = result.ground_truth

        print(f"\n  GROUND TRUTH (Semantic Engine)")
        print(f"  {gt.description}")
        if isinstance(gt.answer, list):
            for rank, (k, v) in enumerate(gt.answer[:5], 1):
                print(f"    #{rank}  {k:<22}  {v:>14,.2f}")
        else:
            print(f"    Exact answer: {_fmt_answer(gt)}")
        print(f"  Latency: {gt.latency_ms}ms | Rows scanned: {gt.rows_scanned:,}")

        print(f"\n  RAG SIMULATION — what the LLM receives vs. correct answer")
        print(f"\n  {'Context':<18} {'Rows':>6} {'Coverage':>10} "
              f"{'Partial sum':>14}  {'Error detectable?'}")
        print(f"  {'-'*18} {'-'*6} {'-'*10} {'-'*14}  {'-'*30}")

        correct = gt.answer if isinstance(gt.answer, float) else gt.answer[0][1]
        for ctx in result.rag_contexts:
            cs      = ctx.confidence_signals
            partial = cs["partial_sum_visible"]
            detect  = cs["detectability_score"].split("—")[0].strip()
            print(
                f"  {'~'+str(ctx.token_estimate//1000)+'K tokens':<18} "
                f"{ctx.rows_retrieved:>6,} "
                f"{ctx.coverage_pct:>9.4f}% "
                f"{partial:>14,.2f}  "
                f"{detect}"
            )

        print(f"\n  Correct answer : {correct:>14,.2f}")
        print(f"  At 8K rows the LLM's partial sum is "
              f"{result.rag_contexts[-1].confidence_signals['partial_sum_visible']:,.2f} "
              f"— confidently presented, structurally wrong.")

    print(f"\n{SEP2}")
    print("  KEY FINDING — ERROR OBSERVABILITY COLLAPSE")
    print(SEP2)
    print("""
  Context size has no bearing on whether RAG can aggregate.
  It only determines how convincing the wrong answer looks.

  At     5 rows: the LLM clearly guesses   — error is obvious
  At 8,000 rows: the LLM sounds authoritative — error is hidden

  The semantic engine scans the full dataset in one deterministic pass.
  No retrieval. No inference. No hallucination. No API call.
  """)


def section_solution(rows) -> None:
    """Part 2: show the router solving the same queries exactly."""

    print(f"\n{SEP2}")
    print("  THE SOLUTION — QUERY ROUTER")
    print("  The router classifies every query before it reaches the pipeline.")
    print("  Computation queries go to the SemanticEngine. Never to RAG.")
    print(SEP2)

    router = QueryRouter(rows)

    demo_queries = [
        "What is the total spend by category?",
        "What is the total spend where amount is greater than 500?",
        "Find transactions from Jennifer Banks",
    ]

    print()
    for query in demo_queries:
        result = router.route(query)
        intent = result.intent
        route_label = "→ SemanticEngine" if intent.is_computation else "→ RAG (safe)"

        print(f"  Query : \"{query}\"")
        print(f"  Route : {intent.route:<14}  {route_label}")
        print(f"  Tier  : {intent.matched_tier}  |  "
              f"matched='{intent.matched_pattern}'  |  "
              f"confidence={intent.confidence:.2f}")

        if intent.is_computation and hasattr(result.answer, 'answer'):
            gt = result.answer
            if isinstance(gt.answer, list):
                top = gt.answer[0]
                q_lower = query.lower()
                is_min = re.search(r"\b(lowest|minimum|least|smallest)\b", q_lower)
                rank_label = "lowest" if is_min else "#1"
                print(f"  Answer: {rank_label} {top[0]:<20} {top[1]:>14,.2f}  "
                      f"({gt.latency_ms}ms | {gt.rows_scanned:,} rows scanned | exact)")
            else:
                parsed_q = result.parsed
                is_ratio = (
                    parsed_q is not None
                    and parsed_q.parse_notes.startswith("ratio query")
                )
                if is_ratio:
                    print(f"  Answer: {gt.answer:>14,.4f}%  "
                          f"({gt.latency_ms}ms | {gt.rows_scanned:,} rows scanned | exact)")
                else:
                    print(f"  Answer: {gt.answer:>14,.2f}  "
                          f"({gt.latency_ms}ms | {gt.rows_scanned:,} rows scanned | exact)")
        else:
            print(f"  Signal: RAG is appropriate — no aggregation required")
        print()

    counts = router.route_counts
    print(f"{SEP}")
    print(f"  Queries routed to SemanticEngine : {counts[Route.COMPUTATION]}")
    print(f"  Queries routed to RAG            : {counts[Route.RETRIEVAL]}")
    print(f"  RAG never sees an aggregation query.")
    print(f"  Error Observability Collapse cannot occur.")
    print(f"{SEP2}\n")


def main() -> None:
    print(f"\n{SEP2}")
    print("  Larger Context Windows Don't Fix RAG — They Make Errors Harder to Detect")
    print("  Across 7 query types on a 100K-row dataset, increasing context size")
    print("  didn't improve accuracy — it made errors harder to detect.")
    print("  Part 1: The Problem  |  Part 2: The Solution")
    print(SEP2)

    rows = load_csv(CSV_PATH, max_rows=100_000)
    print()

    # ── Part 1: the problem ──────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  PART 1 — THE PROBLEM")
    print("  Larger Context Windows Don't Fix RAG — They Make Errors Harder to Detect")
    print("  Across 7 query types on a 100K-row dataset, increasing context size")
    print("  didn't improve accuracy — it made errors harder to detect.")
    print(SEP2)

    section_problem(rows)

    # ── Part 2: the solution ─────────────────────────────────────────────
    section_solution(rows)

    print(f"  Run the full benchmark   :  python context_window_engine.py")
    print(f"  Run the router demo      :  python query_router.py")
    print(f"  Run engine test suite    :  python -m unittest test_engine -v")
    print(f"  Run router test suite    :  python -m unittest test_router -v")
    print(f"  Full dataset (1.29M rows):  python context_window_engine.py --full")
    print()


if __name__ == "__main__":
    main()
