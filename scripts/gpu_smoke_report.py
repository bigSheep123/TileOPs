#!/usr/bin/env python3
"""Generate a GPU smoke markdown report from pytest JUnit XML."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nightly_report import _table_cell, git_commit, testcase_fields, testcase_properties

_PASS = "\u2705"
_FAIL = "\u274c"


def parse_test_xml(path: str) -> list[dict]:
    return [testcase_fields(tc, testcase_properties(tc)) for tc in ET.parse(path).iter("testcase")]


def generate_report(results: list[dict], target: str) -> str:
    total_cases = len(results)
    failed_cases = [result for result in results if result["outcome"] == "failed"]
    passed_count = sum(1 for result in results if result["outcome"] == "passed")
    skipped_count = sum(1 for result in results if result["outcome"] == "skipped")
    failed_ops = list(dict.fromkeys(result["op"] for result in failed_cases if result["op"]))

    correctness = f"{_PASS} Pass" if not failed_cases else f"{_FAIL} {len(failed_cases)} failed"
    failures_ops_str = ", ".join(failed_ops) if failed_ops else "-"
    health = _PASS if not failed_cases else _FAIL

    lines = [
        f"# {health} TileOPs GPU Smoke Report",
        "",
        f"> `{git_commit()}`",
        "",
        "## Summary",
        "",
        "| | |",
        "|---|---|",
        f"| **Correctness** | {correctness} |",
        f"| **gpu-smoke target** | `{target}` |",
        (
            f"| **Gpu-smoke ops number** | {total_cases} "
            f"({passed_count} passed, {skipped_count} skipped) |"
        ),
        f"| **Gpu-smoke Failures** | {len(failed_cases)} |",
        f"| **Failures ops** | {failures_ops_str} |",
        "",
    ]

    if failed_cases:
        lines.extend(
            [
                "## Failure Details",
                "",
                "| Op | Testcase | Failure Reason |",
                "|:---|:---------|:---------------|",
            ]
        )
        for result in failed_cases:
            op = result["op"] or "-"
            testcase = (result["name"] or result["nodeid"]).replace("|", "\\|")
            reason = _table_cell(result["failure_message"] or "No failure message available.")
            lines.append(f"| {op} | {testcase} | {reason} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TileOPs GPU smoke report")
    parser.add_argument("--test-xml", required=True, help="Path to pytest JUnit XML")
    parser.add_argument("--target", required=True, help="Displayed gpu-smoke target label")
    parser.add_argument("--output", required=True, help="Output markdown report path")
    args = parser.parse_args()

    results = parse_test_xml(args.test_xml)
    report = generate_report(results, args.target)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
