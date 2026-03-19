"""
Post-process one rollout JSONL file and compute evaluation metrics.

Metrics per file:
  - success_rate
  - average steps to success
  - std of success indicator (0/1)
  - std of steps to success

Optional:
  Attach config snapshot from config/config.yaml (or a custom path).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def _safe_pstdev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return pstdev(values)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{idx} invalid JSON: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _infer_memory_type(path: Path) -> str:
    name = path.stem
    # expected: rollout_<memory>_<timestamp>
    if name.startswith("rollout_"):
        body = name[len("rollout_") :]
        parts = body.split("_")
        if len(parts) >= 3:
            # strip trailing timestamp chunks
            return "_".join(parts[:-2])
        if len(parts) >= 1:
            return parts[0]
    return "unknown"


def _file_metrics(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    if not rows:
        return {
            "file": str(path),
            "memory_type_from_filename": _infer_memory_type(path),
            "env_name": None,
            "agent_type": None,
            "run_seed": None,
            "episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
            "success_std_episode_level": 0.0,
            "avg_steps_to_success": None,
            "std_steps_to_success": None,
            "avg_steps_all": None,
            "std_steps_all": None,
        }

    success_flags: list[int] = []
    success_steps: list[float] = []
    all_steps: list[float] = []

    for ep in rows:
        reward = float(ep.get("total_reward", 0.0))
        steps = float(ep.get("num_steps", 0))
        success = 1 if reward > 0.0 else 0
        success_flags.append(success)
        all_steps.append(steps)
        if success:
            success_steps.append(steps)

    env_name = rows[0].get("env_name")
    agent_type = rows[0].get("agent_type")
    run_seed = rows[0].get("run_seed")

    return {
        "file": str(path),
        "memory_type_from_filename": _infer_memory_type(path),
        "env_name": env_name,
        "agent_type": agent_type,
        "run_seed": run_seed,
        "episodes": len(rows),
        "successes": sum(success_flags),
        "success_rate": mean(success_flags),
        "success_std_episode_level": _safe_pstdev([float(x) for x in success_flags]),
        "avg_steps_to_success": _safe_mean(success_steps),
        "std_steps_to_success": _safe_pstdev(success_steps),
        "avg_steps_all": _safe_mean(all_steps),
        "std_steps_all": _safe_pstdev(all_steps),
    }


def _maybe_load_config(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        from omegaconf import OmegaConf  # type: ignore

        cfg = OmegaConf.load(str(p))
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:
        # Fallback: include raw text if OmegaConf isn't available or parsing fails.
        try:
            return {"raw_text": p.read_text(encoding="utf-8")}
        except Exception:
            return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process one rollout JSONL file.")
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to one rollout JSONL file.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Output summary JSON path (default: next to input file).",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional output CSV path. If omitted, no CSV is written.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Config to attach in summary (set empty string to skip).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    file_metrics = _file_metrics(input_path)
    config_snapshot = _maybe_load_config(args.config if args.config else None)

    summary = {
        "input_file": str(input_path),
        "metrics": file_metrics,
        "config_path": args.config if args.config else None,
        "config_snapshot": config_snapshot,
    }

    if args.out_json:
        out_json = Path(args.out_json)
    else:
        out_json = input_path.with_name(f"{input_path.stem}_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv: Path | None = None
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "file",
                "memory_type_from_filename",
                "env_name",
                "agent_type",
                "run_seed",
                "episodes",
                "successes",
                "success_rate",
                "success_std_episode_level",
                "avg_steps_to_success",
                "std_steps_to_success",
                "avg_steps_all",
                "std_steps_all",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(file_metrics)

    print(f"Wrote JSON summary: {out_json}")
    if out_csv is not None:
        print(f"Wrote CSV summary : {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
