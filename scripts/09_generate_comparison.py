#!/usr/bin/env python3
"""Step 9: the comparison that proves or disproves the thesis.

Fully implemented (src/eval/compare.py) and runnable today against any two
MetricsSummary objects — including fabricated ones, to sanity-check the
verdict logic before real numbers exist. See tests/test_compare.py.

Usage once Step 5 and Step 8 have produced real MetricsSummary data (loaded
however you choose to persist them — this script assumes you'll adapt the
loading bit to whatever format 05/08 end up writing):
  python scripts/09_generate_comparison.py --ungated <path> --gated <path>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.compare import build_comparison_table, evaluate_thesis, render_markdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ungated", help="Path to Step 5 metrics (format TBD in 05_train_baseline_policy.py)")
    parser.add_argument("--gated", help="Path to Step 8 metrics (format TBD in 08_run_gated_policy.py)")
    args = parser.parse_args()

    if not args.ungated or not args.gated:
        print(
            "No real metrics files yet (Steps 5 and 8 are not run). "
            "src/eval/compare.py is implemented and tested — see "
            "tests/test_compare.py for a working example against synthetic "
            "MetricsSummary data."
        )
        return

    raise NotImplementedError(
        "Loading real MetricsSummary from disk depends on how 05/08 "
        "serialize their results, which doesn't exist yet."
    )


if __name__ == "__main__":
    main()
