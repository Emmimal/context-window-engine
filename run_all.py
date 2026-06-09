"""
run_all.py
----------
Single command that runs the complete project in sequence.

Without CSV (tests + router classification only):
    python run_all.py

With CSV (full benchmark + demo + router with live answers):
    python run_all.py --csv data/credit_card_transactions.csv

Output saved to: run_all_output.txt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

SEP  = "-" * 72
SEP2 = "=" * 72

STEPS_NO_CSV = [
    {
        "label": "Engine test suite (87 tests)",
        "cmd":   [sys.executable, "-m", "unittest", "test_engine", "-v"],
    },
    {
        "label": "Router test suite (72 tests)",
        "cmd":   [sys.executable, "-m", "unittest", "test_router", "-v"],
    },
    {
        "label": "Router classification demo (no CSV needed)",
        "cmd":   [sys.executable, "query_router.py", "--classify-only"],
    },
]

STEPS_WITH_CSV = [
    {
        "label": "Engine test suite (87 tests)",
        "cmd":   [sys.executable, "-m", "unittest", "test_engine", "-v"],
    },
    {
        "label": "Router test suite (72 tests)",
        "cmd":   [sys.executable, "-m", "unittest", "test_router", "-v"],
    },
    {
        "label": "Demo — problem + solution",
        "cmd":   [sys.executable, "demo.py"],
    },
    {
        "label": "Router with live answers",
        "cmd":   [sys.executable, "query_router.py", "--csv", "{csv}"],
    },
    {
        "label": "Full benchmark (100k rows, 7 queries)",
        "cmd":   [sys.executable, "context_window_engine.py", "--csv", "{csv}"],
    },
]


def run_step(label: str, cmd: list[str], output_lines: list[str]) -> bool:
    """Run one step, stream output, return True if passed."""
    header = f"\n{SEP2}\n  STEP: {label}\n{SEP2}\n"
    print(header)
    output_lines.append(header)

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = round(time.perf_counter() - t0, 1)

    print(result.stdout)
    output_lines.append(result.stdout)

    status = "✓  PASSED" if result.returncode == 0 else "✗  FAILED"
    footer = f"\n  {status}  ({elapsed}s)\n"
    print(footer)
    output_lines.append(footer)

    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete context-window-engine project in one command.",
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "credit_card_transactions.csv"),
        help="Path to CSV file (default: data/credit_card_transactions.csv next to this script).",
    )
    parser.add_argument(
        "--output",
        default="run_all_output.txt",
        help="File to save full output (default: run_all_output.txt)",
    )
    args = parser.parse_args()

    output_lines: list[str] = []

    banner = (
        f"\n{SEP2}\n"
        f"  Larger Context Windows Don't Fix RAG — They Make Errors Harder to Detect\n"
        f"  Across 7 query types on a 100K-row dataset, increasing context size\n"
        f"  didn't improve accuracy — it made errors harder to detect.\n"
        f"  {'With CSV: ' + args.csv if args.csv else 'No CSV — tests + router only'}\n"
        f"{SEP2}\n"
    )
    print(banner)
    output_lines.append(banner)

    if args.csv:
        steps = [
            {
                "label": s["label"],
                "cmd": [c.replace("{csv}", args.csv) for c in s["cmd"]],
            }
            for s in STEPS_WITH_CSV
        ]
    else:
        print(f"  CSV not found at: {args.csv}")
        print("  Place your CSV at data/credit_card_transactions.csv to run the full benchmark.")
        print()
        steps = STEPS_NO_CSV

    results: list[tuple[str, bool]] = []

    for step in steps:
        passed = run_step(step["label"], step["cmd"], output_lines)
        results.append((step["label"], passed))

    # Summary
    summary_lines = [f"\n{SEP2}", "  SUMMARY", SEP2]
    all_passed = True
    for label, passed in results:
        icon = "✓" if passed else "✗"
        summary_lines.append(f"  {icon}  {label}")
        if not passed:
            all_passed = False

    total   = len(results)
    n_pass  = sum(1 for _, p in results if p)
    summary_lines.append(f"\n  {n_pass}/{total} steps passed")

    if all_passed:
        summary_lines.append("  All steps passed.")
    else:
        summary_lines.append("  Some steps failed — check output above.")

    summary_lines.append(SEP2 + "\n")
    summary = "\n".join(summary_lines)
    print(summary)
    output_lines.append(summary)

    # Save
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"  Full output saved to: {args.output}")
    except OSError as exc:
        print(f"  WARNING: could not save output: {exc}", file=sys.stderr)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
