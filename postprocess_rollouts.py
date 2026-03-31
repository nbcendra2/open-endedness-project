"""Functionality: Compute success rate and step statistics from one rollout JSONL file

Reads the JSONL written by evaluator.py, calculates per-file metrics
(success_rate, average steps to success, loop rate, truncation rate, blocked-forward rate, etc)
and writes a summary JSON (and optional CSV).
For the config snapshot: prefers the one embedded in the JSONL (episode 0);
falls back to reading config/config.yaml with a warning if not found
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


# Safe wrappers around mean/pstdev that return None for empty lists
# instead of raising StatisticsError

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


def _wilson_ci_95(successes: int, n: int) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion.

    More reliable than the normal approximation for small n or extreme p.
    Returns (lower, upper) bounds.
    """
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p_hat = successes / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file (one JSON object per line) and return a list of dicts"""
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


def is_episode_success(ep: dict[str, Any]) -> bool:
    """Whether an episode counts as solved for success-rate metrics (indicator ``s_i``).

    **Primary rule (current evaluator JSONL):** BabyAI signals task completion with
    ``terminated=True`` and a strictly positive cumulative ``total_reward``.

    **Legacy rows** without a ``terminated`` field keep the old proxy
    ``total_reward > 0`` so existing rollout files still aggregate the same way as
    before when that field was missing.

    If the line already contains an explicit boolean ``success`` (written by the
    evaluator), that value is trusted.
    """
    if isinstance(ep.get("success"), bool):
        return bool(ep["success"])
    reward = float(ep.get("total_reward", 0.0))
    if reward <= 0.0:
        return False
    if "terminated" not in ep:
        return True
    return bool(ep["terminated"])


def episode_environment_steps(ep: dict[str, Any]) -> int:
    """Number of environment steps in one episode (``T_i`` for average steps to success).

    This is how many times the evaluator called ``env.step`` (same as logged
    ``num_steps``). It is **not** an internal gym timestep if that ever differed.

    **Pitfalls handled:**

    - Missing ``num_steps``: fall back to ``len(steps)`` so older or minimal JSONL
      still aggregates.
    - Bad types: coerced; negative values clamped to 0.
    - Do not confuse with ``max_steps_per_episode`` from config: a failed run may
      use fewer steps than the cap; AS only averages over **successful** episodes.
    """
    raw = ep.get("num_steps")
    if raw is not None:
        try:
            n = int(float(raw))
        except (TypeError, ValueError):
            n = 0
        return max(0, n)
    steps = ep.get("steps")
    if isinstance(steps, list):
        return len(steps)
    return 0


# --- Loop rate (sec. 9.3): reproducible heuristic detectors for l_i in LR = (1/N) sum l_i
# Tune these only with care: they trade off false positives vs false negatives.

LOOP_STREAK_MIN = 5
# Same normalized observation + same action_used, repeated this many steps in a row.

LOOP_OSCILLATION_LEN = 6
# Minimum length of A,B,A,B,... with A != B (e.g. turn-left / turn-right).

LOOP_DETECTOR_ID = f"v1_streak{LOOP_STREAK_MIN}_osc{LOOP_OSCILLATION_LEN}"
# Identifies this heuristic; stored in metrics for reproducibility.

BFR_DETECTOR_ID = "v1_go_forward_grid_pos_unchanged"
# Blocked forward: movement_attempt and grid position unchanged (see step ``transition``).


def _norm_obs_text(obs: Any) -> str:
    """Normalize observation text so small whitespace differences do not break streaks."""
    s = str(obs or "")
    return " ".join(s.lower().split())


def _step_action_used(step: dict[str, Any]) -> str:
    """Executed action string (after invalid-action fallback), lowercased."""
    a = step.get("action_used")
    if a is not None and str(a).strip() != "":
        return str(a).strip().lower()
    ao = step.get("agent_output")
    if isinstance(ao, dict) and ao.get("action"):
        return str(ao["action"]).strip().lower()
    return ""


def _loop_streak_same_obs_and_action(steps: list[dict[str, Any]]) -> bool:
    """True if the same (obs, action) pair repeats LOOP_STREAK_MIN times in a row."""
    if len(steps) < LOOP_STREAK_MIN:
        return False
    last_key: tuple[str, str] | None = None
    run = 0
    for s in steps:
        key = (_norm_obs_text(s.get("text_obs")), _step_action_used(s))
        if not key[1]:
            last_key = None
            run = 0
            continue
        if last_key is None:
            last_key = key
            run = 1
            continue
        if key == last_key:
            run += 1
        else:
            last_key = key
            run = 1
        if run >= LOOP_STREAK_MIN:
            return True
    return False


def _loop_two_action_oscillation(steps: list[dict[str, Any]]) -> bool:
    """True if actions contain a length-LOOP_OSCILLATION_LEN window A,B,A,B,... with A != B."""
    actions = [_step_action_used(s) for s in steps]
    n = len(actions)
    if n < LOOP_OSCILLATION_LEN:
        return False
    for i in range(n - LOOP_OSCILLATION_LEN + 1):
        window = actions[i : i + LOOP_OSCILLATION_LEN]
        if not all(window):
            continue
        if window[0] == window[1]:
            continue
        if all(window[j] == window[j % 2] for j in range(LOOP_OSCILLATION_LEN)):
            return True
    return False


def episode_has_loop(ep: dict[str, Any]) -> bool:
    """Episode-level loop flag ``l_i`` for loop rate (sec. 9.3).

    Uses only per-step fields in rollout JSONL (typically ``steps``): **streak**
    (repeated same observation + same executed action) or **oscillation**
    (alternating two actions, e.g. spin in place).

    **Pitfalls:**

    - **Missing / non-list ``steps``:** returns ``False`` (cannot detect; not counted as loop).
    - **False positives:** long corridor with **unchanged** text and repeated ``go forward``
      could trigger the streak rule — rare if descriptions update each step; if your
      obs are very coarse, increase :data:`LOOP_STREAK_MIN`.
    - **False negatives:** clever loops longer than two actions, or progress with changing
      obs, are not covered.
    - **Trust:** if ``has_loop`` is already present on the row (from evaluator), that
      boolean is used so file and metrics stay aligned.
    """
    if isinstance(ep.get("has_loop"), bool):
        return bool(ep["has_loop"])
    steps = ep.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return False
    return _loop_streak_same_obs_and_action(steps) or _loop_two_action_oscillation(steps)


def episode_truncated_budget(ep: dict[str, Any]) -> bool:
    """True if the episode hit a step budget (truncation) rather than ending with task termination.

    Counts Gymnasium ``truncated`` or, when the harness used all ``max_steps`` without
    ``terminated``, the eval budget as truncation (common when TimeLimit is absent).
    """
    if bool(ep.get("truncated")):
        return True
    if bool(ep.get("terminated")):
        return False
    n = int(episode_environment_steps(ep))
    cap = ep.get("max_steps")
    try:
        cap_i = int(cap) if cap is not None else None
    except (TypeError, ValueError):
        cap_i = None
    if cap_i is None:
        return False
    return n >= cap_i


def rollout_bfr_totals(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Return (blocked_forward_steps, forward_attempt_steps) from ``transition`` fields."""
    blocked = 0
    attempt = 0
    for ep in rows:
        steps = ep.get("steps")
        if not isinstance(steps, list):
            continue
        for st in steps:
            if not isinstance(st, dict):
                continue
            tr = st.get("transition")
            if not isinstance(tr, dict):
                continue
            if not tr.get("movement_attempt"):
                continue
            attempt += 1
            if tr.get("movement_blocked"):
                blocked += 1
    return blocked, attempt


def _infer_memory_type(path: Path) -> str:
    """Guess the memory type from the filename pattern rollout_<memory>_<timestamp>

    Example: rollout_trajectory_20260317_081019.jsonl -> "trajectory"
    """
    name = path.stem
    if name.startswith("rollout_"):
        body = name[len("rollout_"):]
        parts = body.split("_")
        if len(parts) >= 3:
            # Last two parts are the timestamp (YYYYMMDD_HHMMSS), everything before is the memory type
            return "_".join(parts[:-2])
        if len(parts) >= 1:
            return parts[0]
    return "unknown"


def _file_metrics_from_rows(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """Compute evaluation metrics from parsed JSONL rows

    Episode success uses :func:`is_episode_success` (terminated + positive reward;
    see docstring there).

    **Average steps to success (sec. 9.2):** with ``T_i`` = :func:`episode_environment_steps`
    and ``s_i`` from :func:`is_episode_success`, AS = (sum_i s_i * T_i) / (sum_i s_i).
    If no episode succeeds, ``avg_steps_to_success`` is ``None`` (denominator 0).

    **Loop rate (sec. 9.3):** ``l_i`` = :func:`episode_has_loop`, then
    LR = (1/N) sum_i l_i. See :data:`LOOP_DETECTOR_ID` for the heuristic version.

    **Truncation rate:** fraction of episodes with :func:`episode_truncated_budget`.

    **Blocked forward rate (BFR):** from per-step ``transition.movement_attempt`` /
    ``movement_blocked``; ratio of blocked ``go forward`` steps to forward attempts.
    ``None`` if no forward attempts. See :data:`BFR_DETECTOR_ID`.
    """
    # Empty file: return zeroed-out metrics
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
            "success_rate_ci95_lower": 0.0,
            "success_rate_ci95_upper": 0.0,
            "success_std_episode_level": 0.0,
            "avg_steps_to_success": None,
            "std_steps_to_success": None,
            "avg_steps_all": None,
            "std_steps_all": None,
            "loop_rate": 0.0,
            "episodes_with_loop": 0,
            "loop_detector": LOOP_DETECTOR_ID,
            "truncation_rate": 0.0,
            "episodes_truncated": 0,
            "blocked_forward_rate": None,
            "blocked_forward_steps": 0,
            "forward_attempt_steps": 0,
            "bfr_detector": BFR_DETECTOR_ID,
        }

    # Collect per-episode numbers (s_i, T_i)
    success_flags: list[int] = []
    success_steps: list[float] = []  # T_i for episodes with s_i = 1
    all_steps: list[float] = []

    for ep in rows:
        t_i = float(episode_environment_steps(ep))
        s_i = 1 if is_episode_success(ep) else 0
        success_flags.append(s_i)
        all_steps.append(t_i)
        if s_i:
            success_steps.append(t_i)

    successes = sum(success_flags)
    n_episodes = len(rows)
    ci_lower, ci_upper = _wilson_ci_95(successes, n_episodes)

    # AS = sum_i (s_i * T_i) / sum_i s_i  (same as mean(success_steps) when successes > 0)
    avg_steps_to_success = (
        sum(success_steps) / successes if successes else None
    )

    loop_flags = [1 if episode_has_loop(ep) else 0 for ep in rows]
    episodes_with_loop = sum(loop_flags)
    loop_rate = mean(loop_flags) if rows else 0.0

    trunc_flags = [1 if episode_truncated_budget(ep) else 0 for ep in rows]
    episodes_truncated = sum(trunc_flags)
    truncation_rate = mean(trunc_flags) if rows else 0.0

    bf_blocked, bf_attempt = rollout_bfr_totals(rows)
    blocked_forward_rate = (
        bf_blocked / bf_attempt if bf_attempt else None
    )

    # Metadata from the first episode (same for all episodes in one run)
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
        "successes": successes,
        "success_rate": mean(success_flags),
        "success_rate_ci95_lower": ci_lower,
        "success_rate_ci95_upper": ci_upper,
        "success_std_episode_level": _safe_pstdev([float(x) for x in success_flags]),
        "avg_steps_to_success": avg_steps_to_success,
        "std_steps_to_success": _safe_pstdev(success_steps),
        "avg_steps_all": _safe_mean(all_steps),
        "std_steps_all": _safe_pstdev(all_steps),
        "loop_rate": loop_rate,
        "episodes_with_loop": episodes_with_loop,
        "loop_detector": LOOP_DETECTOR_ID,
        "truncation_rate": truncation_rate,
        "episodes_truncated": episodes_truncated,
        "blocked_forward_rate": blocked_forward_rate,
        "blocked_forward_steps": bf_blocked,
        "forward_attempt_steps": bf_attempt,
        "bfr_detector": BFR_DETECTOR_ID,
    }

    # Aggregate fade_enriched_history memory diagnostics (if present in JSONL)
    memory_stat_keys = [
        "active_facts", "dormant_facts", "fused_facts",
        "reactivations", "priming_boosts",
        "tag_triggers_fired", "tag_shields_expired",
        "contradictions", "compatible_resolved",
        "subsume_resolved", "fusions_performed",
    ]
    episodes_with_stats = [
        ep["memory_stats"] for ep in rows
        if isinstance(ep.get("memory_stats"), dict)
    ]
    if episodes_with_stats:
        fade_agg: dict[str, Any] = {}
        for key in memory_stat_keys:
            vals = [float(ms.get(key, 0)) for ms in episodes_with_stats]
            fade_agg[f"avg_{key}"] = _safe_mean(vals)
            fade_agg[f"std_{key}"] = _safe_pstdev(vals)
        fade_agg["episodes_with_memory_stats"] = len(episodes_with_stats)
        result["fade_memory_stats"] = fade_agg

    return result


def _embedded_config_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Look for config_snapshot inside the JSONL (evaluator embeds it on episode 0)

    This is the config that was active when the experiment ran,
    so it always matches the results in this file
    """
    for row in rows:
        snap = row.get("config_snapshot")
        if isinstance(snap, dict) and snap:
            return snap
    return None


def _maybe_load_config(path: str | None) -> dict[str, Any] | None:
    """Try to load a YAML config file from disk as a dict

    Used as fallback when the JSONL has no embedded config_snapshot (old rollouts)
    """
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
        help="Fallback YAML if rollout has no embedded config_snapshot (set empty string to skip).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    # Step 1: read all episode lines from the JSONL
    rows = _load_jsonl(input_path)

    # Step 2: compute metrics (success rate, steps, etc)
    file_metrics = _file_metrics_from_rows(rows, input_path)

    # Step 3: get config - prefer the one stored inside the JSONL (matches the run);
    # if not found (old format), fall back to reading config.yaml from disk
    embedded = _embedded_config_from_rows(rows)
    if embedded is not None:
        config_snapshot = embedded
        config_snapshot_source = "rollout_jsonl"
    else:
        config_snapshot = _maybe_load_config(args.config if args.config else None)
        config_snapshot_source = "yaml_file" if config_snapshot is not None else "none"

    # Step 4: build the summary dict
    summary = {
        "input_file": str(input_path),
        "metrics": file_metrics,
        "config_path": args.config if args.config else None,
        "config_snapshot": config_snapshot,
        "config_snapshot_source": config_snapshot_source,  # "rollout_jsonl" or "yaml_file" or "none"
    }

    # Warn if config comes from the YAML file (may not match the run, see module docstring)
    if config_snapshot_source == "yaml_file":
        print(
            "warning: this rollout has no embedded config_snapshot (old format). "
            "config_snapshot in the summary is read from the YAML file now, "
            "which may not be the same settings as when the run was executed. "
            "Re-run the evaluator to get rollouts that embed config on episode 0.",
            file=sys.stderr,
        )

    # Step 5: write summary JSON
    if args.out_json:
        out_json = Path(args.out_json)
    else:
        out_json = input_path.with_name(f"{input_path.stem}_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Step 6 (optional): write a one-row CSV with the same metrics
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
                "success_rate_ci95_lower",
                "success_rate_ci95_upper",
                "success_std_episode_level",
                "avg_steps_to_success",
                "std_steps_to_success",
                "avg_steps_all",
                "std_steps_all",
                "loop_rate",
                "episodes_with_loop",
                "loop_detector",
                "truncation_rate",
                "episodes_truncated",
                "blocked_forward_rate",
                "blocked_forward_steps",
                "forward_attempt_steps",
                "bfr_detector",
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
