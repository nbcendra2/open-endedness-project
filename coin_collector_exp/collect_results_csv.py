#!/usr/bin/env python
"""Scan *_summary.json files and produce a single CSV results table.

Usage:
    python collect_results_csv.py                          # scans results/ and runs/
    python collect_results_csv.py results/                 # scan one directory
    python collect_results_csv.py results/ runs/ -o results/all_results.csv
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

FIELDNAMES = [
    "env_name",
    "level",
    "provider",
    "model",
    "memory_type",
    "seed",
    "episodes",
    "successes",
    "success_rate",
    "ci95_lower",
    "ci95_upper",
    "avg_steps_all",
    "std_steps_all",
    "avg_steps_to_success",
    "std_steps_to_success",
    "loop_rate",
    "truncation_rate",
    "blocked_forward_rate",
    "blocked_forward_steps",
    "forward_attempt_steps",
    "episodes_with_loop",
    "episodes_truncated",
    "rollout_file",
    "summary_file",
]


def _extract_level(env_name: str) -> str:
    """Pull a level tag like 'L5' from env names such as 'tw-coin_collector-L5'."""
    m = re.search(r"-L(\d+)", env_name)
    if m:
        return f"L{m.group(1)}"
    # BabyAI: use the short name as the level identifier
    return env_name.replace("BabyAI-", "").replace("-v0", "")


def _extract_row(summary: dict, summary_path: Path) -> dict:
    """Flatten a summary JSON into one CSV-friendly row."""
    m = summary.get("metrics", {})
    cfg = summary.get("config_snapshot") or {}
    env_cfg = cfg.get("env", {})
    llm_cfg = cfg.get("llm", {})
    agent_cfg = cfg.get("agent", {}).get("params", {})
    eval_cfg = cfg.get("eval", {})

    env_name = env_cfg.get("name", "")

    return {
        "env_name": env_name,
        "level": _extract_level(env_name),
        "provider": llm_cfg.get("provider", ""),
        "model": llm_cfg.get("name", ""),
        "memory_type": agent_cfg.get("memory_type", ""),
        "seed": eval_cfg.get("seed", ""),
        "episodes": m.get("episodes", ""),
        "successes": m.get("successes", ""),
        "success_rate": m.get("success_rate", ""),
        "ci95_lower": m.get("success_rate_ci95_lower", ""),
        "ci95_upper": m.get("success_rate_ci95_upper", ""),
        "avg_steps_all": m.get("avg_steps_all", ""),
        "std_steps_all": m.get("std_steps_all", ""),
        "avg_steps_to_success": m.get("avg_steps_to_success", ""),
        "std_steps_to_success": m.get("std_steps_to_success", ""),
        "loop_rate": m.get("loop_rate", ""),
        "truncation_rate": m.get("truncation_rate", ""),
        "blocked_forward_rate": m.get("blocked_forward_rate", ""),
        "blocked_forward_steps": m.get("blocked_forward_steps", ""),
        "forward_attempt_steps": m.get("forward_attempt_steps", ""),
        "episodes_with_loop": m.get("episodes_with_loop", ""),
        "episodes_truncated": m.get("episodes_truncated", ""),
        "rollout_file": summary.get("input_file", ""),
        "summary_file": str(summary_path),
    }


def collect_summaries(directory: Path) -> list[dict]:
    """Read every *_summary.json under *directory*, return flat row dicts."""
    rows = []
    for p in sorted(directory.rglob("*_summary.json")):
        try:
            summary = json.loads(p.read_text(encoding="utf-8"))
            rows.append(_extract_row(summary, p))
        except Exception as exc:
            print(f"  ⚠ skipping {p}: {exc}")
    return rows


def write_csv(rows: list[dict], out_path: Path) -> Path:
    """Write rows to CSV, creating parent directories as needed.

    Rows are sorted by env_name → provider → memory_type for readability.
    Returns the resolved output path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(
        key=lambda r: (
            r.get("env_name", ""),
            r.get("provider", ""),
            r.get("memory_type", ""),
        )
    )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Collect summary JSONs into a CSV")
    parser.add_argument(
        "dirs",
        nargs="*",
        default=["results", "runs"],
        help="Directories to scan (default: results/ and runs/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: results/all_results_<timestamp>.csv)",
    )
    args = parser.parse_args()

    all_rows: list[dict] = []
    for d in args.dirs:
        p = Path(d)
        if p.is_dir():
            print(f"Scanning {p}/ ...")
            found = collect_summaries(p)
            print(f"  Found {len(found)} summary file(s)")
            all_rows.extend(found)
        else:
            print(f"  Skipping {d} (not a directory)")

    if not all_rows:
        print("No summary files found.")
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"results/all_results_{ts}.csv")

    out = write_csv(all_rows, out_path)
    print(f"\nWrote {len(all_rows)} rows → {out}")


if __name__ == "__main__":
    main()