#!/usr/bin/env python3
"""TextWorld TreasureHunter evaluator — standalone, additive script.

Runs the adaptive-memento agent on TextWorld TreasureHunter games at
easy / medium / hard difficulties.  Produces JSONL rollouts and summary
JSON files whose schemas mirror evaluator.py + postprocess_rollouts.py.

Platform notes:
    - Linux x86_64:       Best supported. pip install textworld==1.7.0
    - macOS x86_64:       Works with pre-built wheel.
    - macOS Apple Silicon: No ARM64 wheel. Use Rosetta:
                             arch -x86_64 pip install textworld
    - Linux ARM64:        Not supported (no Inform7 ARM binaries).
    - Windows:            Not supported. Use WSL or Docker:
                             docker pull marccote19/textworld

Install:  pip install textworld   (requires network — downloads Inform7 at install)

Usage examples:
    python evaluator_textworld.py --difficulty easy --num-episodes 5
    python evaluator_textworld.py --difficulty all
    python evaluator_textworld.py --difficulty hard --max-steps 200 --num-episodes 20
    python evaluator_textworld.py --batch
    python evaluator_textworld.py --batch --difficulty easy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

# ── Platform check & graceful dependency checks ────────────────────────────

import platform

def _platform_warning() -> Optional[str]:
    """Return a warning string if the current platform has known TextWorld issues."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        return (
            "TextWorld does not support Windows natively.\n"
            "  Options:\n"
            "    1. WSL:    wsl --install, then install inside Ubuntu\n"
            "    2. Docker: docker pull marccote19/textworld"
        )
    if system == "Darwin" and machine == "arm64":
        return (
            "TextWorld has no ARM64 (Apple Silicon) wheel.\n"
            "  Install under Rosetta 2:\n"
            "    arch -x86_64 pip install textworld\n"
            "  Run this script under Rosetta:\n"
            "    arch -x86_64 python evaluator_textworld.py ..."
        )
    if system == "Linux" and machine == "aarch64":
        return (
            "TextWorld does not ship Inform7 binaries for Linux ARM64.\n"
            "  Use an x86_64 machine, or run under QEMU / Docker x86_64 emulation."
        )
    return None

_MISSING: List[str] = []

_pw = _platform_warning()
if _pw:
    print(f"WARNING (platform): {_pw}\n", file=sys.stderr)

try:
    import textworld
    import textworld.gym
except ImportError:
    _MISSING.append("textworld  →  pip install textworld  (requires network for Inform7 download)")

try:
    from textworld.challenges import treasure_hunter as _tw_th
except ImportError:
    if "textworld" not in [m.split()[0] for m in _MISSING]:
        _MISSING.append(
            "textworld.challenges.treasure_hunter not found; "
            "you may need a newer textworld:  pip install 'textworld>=1.5'"
        )

try:
    from omegaconf import OmegaConf, DictConfig
except ImportError:
    _MISSING.append("omegaconf  →  pip install omegaconf")

try:
    from tqdm import tqdm
except ImportError:
    _MISSING.append("tqdm  →  pip install tqdm")

if _MISSING:
    print("ERROR: missing dependencies:\n  " + "\n  ".join(_MISSING), file=sys.stderr)
    if _pw:
        print(f"\nPlatform note:\n  {_pw}", file=sys.stderr)
    sys.exit(1)

# Repo imports (read-only — no repo files are modified)
from agent import build_agent
from environments.env_wrapper import EnvState, StepResult
from postprocess_rollouts import (
    is_episode_success,
    episode_has_loop,
    episode_environment_steps,
    episode_truncated_budget,
    rollout_bfr_totals,
    _wilson_ci_95,
    _safe_mean,
    _safe_pstdev,
    LOOP_DETECTOR_ID,
    BFR_DETECTOR_ID,
)

# ── Difficulty → TreasureHunter level mapping ───────────────────────────────
# Level (1-30) controls room count, quest length, and object density.
# Representative levels chosen per bucket; override via CLI if needed.

DIFFICULTY_MAP: Dict[str, Dict[str, Any]] = {
    "easy":   {"level": 10,  "default_max_steps": 10,  "default_episodes": 2},
    "medium": {"level": 15, "default_max_steps": 20, "default_episodes": 2},
    "hard":   {"level": 20, "default_max_steps": 20, "default_episodes": 2},
}
#hard usually 25 above.


# ── Helpers ─────────────────────────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def _mean_list(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _batch_eta_seconds(finished_durations: List[float], remaining_runs: int) -> Optional[float]:
    if remaining_runs <= 0 or not finished_durations:
        return None
    return _mean_list(finished_durations) * remaining_runs


def _print_batch_progress_line(
    batch_t0: float, run_durations: List[float], run_idx: int, total: int
) -> None:
    """After each sub-run: wall-clock elapsed + ETA from mean duration x remaining."""
    elapsed_batch = time.perf_counter() - batch_t0
    remaining = total - run_idx
    eta_s = _batch_eta_seconds(run_durations, remaining)
    parts = [f"elapsed {_format_duration(elapsed_batch)}"]
    if eta_s is not None:
        finish = datetime.now() + timedelta(seconds=eta_s)
        parts.append(f"~{_format_duration(eta_s)} left (est.)")
        parts.append(f"ETA ~{finish.strftime('%H:%M:%S')}")
    print(f"    Batch: {'  |  '.join(parts)}")


def _unwrap(val: Any) -> Any:
    """TextWorld gym sometimes wraps scalars in single-element lists (batch API)."""
    if isinstance(val, list) and len(val) == 1:
        return val[0]
    return val


# ── Game generation ─────────────────────────────────────────────────────────

def _generate_game(level: int, seed: int, tmp_dir: str) -> str:
    """Generate a TreasureHunter game file. Returns path to .z8/.ulx file.

    treasure_hunter.make() returns a Game object (not a file).
    We then compile it to a playable game file via textworld.generator.compile_game().
    """
    options = textworld.GameOptions()
    options.seeds = seed
    options.path = os.path.join(tmp_dir, f"tw_th_L{level}_s{seed}")

    try:
        game = _tw_th.make({"level": level}, options)
    except Exception as exc:
        raise RuntimeError(
            f"Could not generate TreasureHunter game (level={level}).\n"
            f"Ensure textworld is properly installed: pip install 'textworld>=1.5'\n"
            f"Original error: {exc}"
        ) from exc

    game_file = textworld.generator.compile_game(game, options)
    return game_file


def _make_tw_env(game_file: str):
    """Create a TextWorld gym env from a game file."""
    request_infos = textworld.EnvInfos(
        description=True,
        inventory=True,
        admissible_commands=True,
        objective=True,
        max_score=True,
        won=True,
        lost=True,
    )
    env_id = textworld.gym.register_game(
        game_file,
        request_infos=request_infos,
        max_episode_steps=0,  # 0 = no built-in limit; we manage truncation
    )
    return textworld.gym.make(env_id)


# ── TextWorld ↔ agent interface adapters ────────────────────────────────────

def _tw_text_obs(obs: str, infos: dict) -> str:
    """Combine room description + inventory into one text block."""
    parts = []
    raw_obs = _unwrap(obs) if isinstance(obs, list) else obs
    if raw_obs and str(raw_obs).strip():
        parts.append(str(raw_obs).strip())
    inv = _unwrap(infos.get("inventory", ""))
    if inv and str(inv).strip():
        parts.append(f"Inventory: {str(inv).strip()}")
    return "\n".join(parts) if parts else "(no observation)"


def _tw_to_env_state(obs: Any, infos: dict) -> EnvState:
    """Convert TextWorld obs/infos → EnvState consumed by the agent."""
    mission = _unwrap(infos.get("objective", "Find and take the treasure."))
    mission = str(mission) if mission else "Find and take the treasure."
    admissible = infos.get("admissible_commands", [])
    admissible = _unwrap(admissible) if admissible else []
    if not isinstance(admissible, list):
        admissible = [str(admissible)]
    return EnvState(
        mission=mission,
        text_obs=_tw_text_obs(obs, infos),
        valid_actions=list(admissible),
        raw_obs={"text": str(_unwrap(obs) if isinstance(obs, list) else obs)},
        raw_info=infos,
    )


def _build_system_prompt(objective: str) -> str:
    """System prompt analogous to BabyAI's get_instruction_prompt."""
    return (
        f"You are playing a text adventure game (TextWorld TreasureHunter).\n"
        f"Your objective: {objective}\n\n"
        f"At each step you receive a text observation and a list of valid actions.\n"
        f"Choose EXACTLY one action from the valid actions list.\n"
        f'Respond with a JSON object: {{"action": "<chosen action>", '
        f'"reason": "<brief, 10 words max>"}}\n\n'
        f"Tips:\n"
        f"- Explore rooms by going through doors/exits.\n"
        f"- Pick up objects that match your objective.\n"
        f"- Use 'examine' and 'look' to gather information.\n"
        f"- Common actions: go north/south/east/west, take X, open X, look, inventory."
    )


def _validate_tw_action(
    proposed: str,
    valid_actions: List[str],
    invalid_action_mode: str,
) -> Tuple[str, bool, str]:
    """Validate action. Fallback picks 'look' or first admissible command."""
    p = proposed.strip()
    lower_map = {a.lower(): a for a in valid_actions}
    if p.lower() in lower_map:
        return lower_map[p.lower()], True, ""
    if invalid_action_mode == "strict":
        raise ValueError(f"Invalid action '{proposed}'. Valid: {valid_actions}")
    # Prefer 'look' as safe fallback, else first admissible
    if "look" in lower_map:
        return lower_map["look"], False, f"invalid_action:{proposed}"
    if valid_actions:
        return valid_actions[0], False, f"invalid_action:{proposed}"
    return proposed, False, f"invalid_action:{proposed},no_valid_actions"


# ── TextWorld step helper ───────────────────────────────────────────────────

def _tw_step(tw_env, action: str) -> Tuple[Any, float, bool, bool, dict]:
    """Call tw_env.step and normalise to (obs, reward_delta, terminated, truncated, infos).

    TextWorld gym may return 4-tuple (old gym: obs, cumulative_score, done, infos)
    or 5-tuple (gymnasium: obs, reward, terminated, truncated, infos).
    We detect which and return a uniform 5-tuple with per-step reward.
    """
    out = tw_env.step(action)
    if len(out) == 5:
        # gymnasium-style
        obs, reward, term, trunc, infos = out
        return obs, float(reward), bool(term), bool(trunc), infos
    elif len(out) == 4:
        # old gym: score is cumulative total
        obs, score, done, infos = out
        return obs, float(score), bool(done), False, infos
    else:
        raise RuntimeError(f"Unexpected step() output with {len(out)} elements")


def _tw_reset(tw_env) -> Tuple[Any, dict]:
    """Call tw_env.reset and normalise to (obs, infos)."""
    out = tw_env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out
    # Some old textworld versions return just obs
    return out, {}


# ── Episode runner ──────────────────────────────────────────────────────────

def run_episode(
    agent: Any,
    tw_env: Any,
    episode_idx: int,
    seed: int,
    max_steps: int,
    env_name: str,
    invalid_action_mode: str,
) -> Dict[str, Any]:
    """Run one TextWorld episode. Returns dict matching evaluator.py schema."""
    obs, infos = _tw_reset(tw_env)
    state = _tw_to_env_state(obs, infos)
    system_prompt = _build_system_prompt(state.mission)

    agent.start_episode(
        episode_id=episode_idx,
        mission=state.mission,
        seed=seed + episode_idx,
        system_prompt=system_prompt,
        env_name=env_name,
    )

    steps_log: List[Dict[str, Any]] = []
    total_reward = 0.0
    # Track cumulative score for old-gym API (score is cumulative, not delta)
    prev_cumulative = 0.0
    is_old_gym = False  # detected on first step
    terminated = False
    truncated = False

    for t in range(max_steps):
        out = agent.act(
            text_obs=state.text_obs,
            mission=state.mission,
            valid_actions=state.valid_actions,
            step_idx=t,
        )
        proposed_action = str(out.get("action", ""))

        action_used, was_valid, reason = _validate_tw_action(
            proposed_action, state.valid_actions, invalid_action_mode,
        )

        # Step the TextWorld env
        step_out = tw_env.step(action_used)

        if len(step_out) == 5:
            # gymnasium-style: (obs, reward, terminated, truncated, infos)
            new_obs, step_reward, step_term, step_trunc, new_infos = step_out
            step_reward = float(step_reward)
            done = bool(step_term) or bool(step_trunc)
        elif len(step_out) == 4:
            # old gym: (obs, cumulative_score, done, infos) — compute delta
            new_obs, cumulative_score, done, new_infos = step_out
            cumulative_score = float(cumulative_score)
            step_reward = cumulative_score - prev_cumulative
            prev_cumulative = cumulative_score
            step_term = done
            step_trunc = False
        else:
            raise RuntimeError(f"Unexpected step() output: {len(step_out)} elements")

        new_state = _tw_to_env_state(new_obs, new_infos)

        won = bool(_unwrap(new_infos.get("won", False)))
        lost = bool(_unwrap(new_infos.get("lost", False)))

        # Map TextWorld done semantics → terminated / truncated
        step_terminated = bool(done and (won or lost))
        step_truncated = bool(done and not won and not lost)

        step_result = StepResult(
            state=new_state,
            reward=step_reward,
            terminated=step_terminated,
            truncated=step_truncated,
            action_used=action_used,
            action_was_valid=was_valid,
            reason=reason,
        )

        agent.observe_step(
            step_idx=t,
            prev_text_obs=state.text_obs,
            action=proposed_action,
            step_result=step_result,
        )

        total_reward += step_reward

        # TextWorld has no grid → transition fields are always n/a
        steps_log.append({
            "episode": episode_idx,
            "step_idx": t,
            "mission": state.mission,
            "text_obs": state.text_obs,
            "valid_actions": state.valid_actions,
            "agent_output": {
                "action": proposed_action,
                "reason": out.get("reason", ""),
            },
            "action_used": action_used,
            "action_was_valid": was_valid,
            "reward": step_reward,
            "terminated": step_terminated,
            "truncated": step_truncated,
            "env_reason": reason,
            "transition": {
                "movement_attempt": False,
                "movement_blocked": False,
            },
        })

        state = new_state
        terminated = step_terminated
        truncated = step_truncated
        if done:
            break

    # Budget exhaustion → truncated
    if not terminated and not truncated and len(steps_log) >= max_steps:
        truncated = True

    agent.end_episode(total_reward=total_reward, terminated=terminated)

    ep_result: Dict[str, Any] = {
        "episode": episode_idx,
        "seed": seed + episode_idx,
        "num_steps": len(steps_log),
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "steps": steps_log,
    }

    # Fade memory diagnostics (mirrors evaluator.py lines 291-305)
    memory_type = getattr(agent, "memory_type", "")
    if memory_type == "fade_enriched_history":
        buf = getattr(agent, "_enriched_buffer", [])
        ep_result["memory_stats"] = {
            "active_facts": sum(1 for f in buf if f.get("state") == "active"),
            "dormant_facts": sum(1 for f in buf if f.get("state") == "dormant"),
            "fused_facts": sum(1 for f in buf if f.get("fused_from", 0) > 0),
            "reactivations": agent._fade_reactivation_count,
            "priming_boosts": agent._fade_priming_count,
            "tag_triggers_fired": agent._fade_tag_trigger_count,
            "tag_shields_expired": agent._fade_tag_expire_count,
            "contradictions": agent._fade_contradiction_count,
            "compatible_resolved": agent._fade_compatible_count,
            "subsume_resolved": agent._fade_subsume_count,
            "fusions_performed": agent._fade_fusion_count,
        }

    ep_result["success"] = is_episode_success(ep_result)
    ep_result["has_loop"] = episode_has_loop(ep_result)
    return ep_result


# ── Metrics (delegates to postprocess_rollouts logic) ───────────────────────

def _compute_metrics(rows: List[Dict], path: Path) -> Dict[str, Any]:
    """Compute metrics using the exact same logic as postprocess_rollouts._file_metrics_from_rows.

    blocked_forward_rate is always null for TextWorld (no grid movement).
    """
    if not rows:
        return {
            "file": str(path),
            "env_name": None, "agent_type": None, "run_seed": None,
            "episodes": 0, "successes": 0,
            "success_rate": 0.0,
            "success_rate_ci95_lower": 0.0, "success_rate_ci95_upper": 0.0,
            "success_std_episode_level": 0.0,
            "avg_steps_to_success": None, "std_steps_to_success": None,
            "avg_steps_all": None, "std_steps_all": None,
            "loop_rate": 0.0, "episodes_with_loop": 0, "loop_detector": LOOP_DETECTOR_ID,
            "truncation_rate": 0.0, "episodes_truncated": 0,
            "blocked_forward_rate": None,
            "blocked_forward_rate_explanation": "Not applicable: TextWorld has no grid movement.",
            "blocked_forward_steps": 0, "forward_attempt_steps": 0,
            "bfr_detector": BFR_DETECTOR_ID,
        }

    success_flags: List[int] = []
    success_steps: List[float] = []
    all_steps: List[float] = []

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

    avg_steps_to_success = sum(success_steps) / successes if successes else None

    loop_flags = [1 if episode_has_loop(ep) else 0 for ep in rows]
    trunc_flags = [1 if episode_truncated_budget(ep) else 0 for ep in rows]

    bf_blocked, bf_attempt = rollout_bfr_totals(rows)

    env_name = rows[0].get("env_name")
    agent_type = rows[0].get("agent_type")
    run_seed = rows[0].get("run_seed")

    result: Dict[str, Any] = {
        "file": str(path),
        "env_name": env_name,
        "agent_type": agent_type,
        "run_seed": run_seed,
        "episodes": n_episodes,
        "successes": successes,
        "success_rate": mean(success_flags),
        "success_rate_ci95_lower": ci_lower,
        "success_rate_ci95_upper": ci_upper,
        "success_std_episode_level": _safe_pstdev([float(x) for x in success_flags]),
        "avg_steps_to_success": avg_steps_to_success,
        "std_steps_to_success": _safe_pstdev(success_steps),
        "avg_steps_all": _safe_mean(all_steps),
        "std_steps_all": _safe_pstdev(all_steps),
        "loop_rate": mean(loop_flags) if rows else 0.0,
        "episodes_with_loop": sum(loop_flags),
        "loop_detector": LOOP_DETECTOR_ID,
        "truncation_rate": mean(trunc_flags) if rows else 0.0,
        "episodes_truncated": sum(trunc_flags),
        # BFR is structurally n/a for TextWorld (no grid, no "go forward")
        "blocked_forward_rate": None,
        "blocked_forward_rate_explanation": (
            "Not applicable: TextWorld has no grid movement."
            if bf_attempt == 0 else None
        ),
        "blocked_forward_steps": bf_blocked,
        "forward_attempt_steps": bf_attempt,
        "bfr_detector": BFR_DETECTOR_ID,
    }

    # Fade memory stats (same aggregation as postprocess_rollouts)
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
        fade_agg: Dict[str, Any] = {}
        for key in memory_stat_keys:
            vals = [float(ms.get(key, 0)) for ms in episodes_with_stats]
            fade_agg[f"avg_{key}"] = _safe_mean(vals)
            fade_agg[f"std_{key}"] = _safe_pstdev(vals)
        fade_agg["episodes_with_memory_stats"] = len(episodes_with_stats)
        result["fade_memory_stats"] = fade_agg

    return result


# ── Console output ──────────────────────────────────────────────────────────

def _print_metrics(m: Dict[str, Any]) -> None:
    """Short console summary matching evaluator.py style."""
    n = int(m.get("episodes") or 0)
    s = int(m.get("successes") or 0)
    sr = float(m.get("success_rate") or 0.0)
    ci_lo, ci_hi = m.get("success_rate_ci95_lower"), m.get("success_rate_ci95_upper")
    ci_str = f"  95% CI [{ci_lo:.1%}, {ci_hi:.1%}]" if ci_lo is not None and ci_hi is not None else ""
    print(f"Metrics: success_rate={sr:.1%} ({s}/{n} episodes){ci_str}")

    lr = m.get("loop_rate")
    if lr is not None and n > 0:
        print(f"         loop_rate={float(lr):.1%} ({int(m.get('episodes_with_loop', 0))}/{n} episodes)")
    tr = m.get("truncation_rate")
    if tr is not None and n > 0:
        print(f"         truncation_rate={float(tr):.1%} ({int(m.get('episodes_truncated', 0))}/{n} episodes)")

    explanation = m.get("blocked_forward_rate_explanation")
    if explanation:
        print(f"         blocked_forward_rate=null ({explanation})")

    avg_all = m.get("avg_steps_all")
    if avg_all is not None:
        print(f"         avg_steps_all={avg_all:.2f} (std={m.get('std_steps_all')})")
    avg_ok = m.get("avg_steps_to_success")
    if avg_ok is not None:
        print(f"         avg_steps_to_success={avg_ok:.2f} (std={m.get('std_steps_to_success')})")
    elif n > 0:
        print("         avg_steps_to_success=n/a (no successful episodes)")

    fade = m.get("fade_memory_stats")
    if isinstance(fade, dict):
        print("         --- fade-enriched memory stats (per-episode avg) ---")
        for key in ["active_facts", "dormant_facts", "fused_facts",
                     "reactivations", "priming_boosts", "contradictions",
                     "fusions_performed", "tag_triggers_fired"]:
            avg = fade.get(f"avg_{key}")
            std = fade.get(f"std_{key}")
            if avg is not None:
                print(f"         {key}: avg={avg:.2f} (std={std:.2f})")


# ── Per-difficulty runner ───────────────────────────────────────────────────

def _run_difficulty(
    difficulty: str,
    cfg: DictConfig,
    num_episodes: Optional[int],
    max_steps: Optional[int],
    out_base: Optional[str],
    quiet: bool = False,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Run all episodes for one difficulty. Returns (rollout_path, metrics) or (None, None).

    quiet=True suppresses banners and detailed metrics (used by batch mode).
    """
    settings = DIFFICULTY_MAP[difficulty]
    level = settings["level"]
    # Priority: CLI --num-episodes > config eval.num_episodes > DIFFICULTY_MAP default
    cfg_episodes = int(getattr(cfg.eval, "num_episodes", 0)) or None
    cfg_max_steps = int(getattr(cfg.eval, "max_steps_per_episode", 0)) or None
    n_ep = num_episodes or cfg_episodes or settings["default_episodes"]
    m_steps = max_steps or cfg_max_steps or settings["default_max_steps"]
    seed = int(cfg.eval.seed)
    invalid_action_mode = str(cfg.env.invalid_action_mode)
    env_name = f"tw-treasure_hunter-L{level}-{difficulty}"

    if not quiet:
        print(f"\n{'─'*60}")
        print(f"  Difficulty: {difficulty}  (TreasureHunter level={level})")
        print(f"  Episodes: {n_ep}, Max steps: {m_steps}, Seed: {seed}")
        print(f"{'─'*60}")

    agent = build_agent(config=cfg, system_prompt=None)
    memory_type = str(
        getattr(getattr(cfg.agent, "params", {}), "memory_type", "baseline")
    ).lower()

    # Use CLI --out-json if provided, else config's eval.out_json directory, else "runs"
    if out_base:
        out_dir = out_base
    else:
        cfg_out = str(getattr(cfg.eval, "out_json", "runs/rollout.jsonl"))
        out_dir = os.path.dirname(cfg_out) or "runs"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"rollout_tw_{difficulty}_{memory_type}_{ts}.jsonl"

    config_snapshot = OmegaConf.to_container(cfg, resolve=True)
    tmp_dir = tempfile.mkdtemp(prefix=f"tw_{difficulty}_")

    try:
        rows: List[Dict[str, Any]] = []

        with open(out_path, "w", encoding="utf-8") as f:
            for ep_idx in tqdm(range(n_ep), desc=f"{difficulty}", total=n_ep):
                ep_seed = seed + ep_idx

                # Generate a game; retry with offset seeds if the map can't support the quest
                game_file = None
                for attempt in range(10):
                    try:
                        game_file = _generate_game(level, ep_seed + attempt * 1000, tmp_dir)
                        break
                    except Exception:
                        continue
                if game_file is None:
                    print(f"\n  WARNING: game generation failed for ep {ep_idx} after 10 retries")
                    continue

                tw_env = _make_tw_env(game_file)
                try:
                    episode_result = run_episode(
                        agent=agent,
                        tw_env=tw_env,
                        episode_idx=ep_idx,
                        seed=seed,
                        max_steps=m_steps,
                        env_name=env_name,
                        invalid_action_mode=invalid_action_mode,
                    )
                finally:
                    tw_env.close()

                line = {
                    "env_name": env_name,
                    "agent_type": memory_type,
                    "run_seed": seed,
                    "max_steps": m_steps,
                    "num_episodes": n_ep,
                    **episode_result,
                }
                if len(rows) == 0:  # first successful episode gets config
                    line["config_snapshot"] = config_snapshot
                json.dump(line, f, ensure_ascii=False)
                f.write("\n")
                rows.append(line)

        if not rows:
            print(f"\n  WARNING: no episodes completed for {difficulty}, removing empty file")
            out_path.unlink(missing_ok=True)
            return None, None

        metrics = _compute_metrics(rows, out_path)

        summary = {
            "input_file": str(out_path),
            "difficulty": difficulty,
            "treasure_hunter_level": level,
            "metrics": metrics,
            "config_snapshot": config_snapshot,
            "config_snapshot_source": "rollout_jsonl",
        }
        summary_path = out_path.with_name(f"{out_path.stem}_summary.json")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if not quiet:
            print(f"Wrote rollout: {out_path}")
            print(f"Wrote summary: {summary_path}")
            _print_metrics(metrics)
        return out_path, metrics

    except Exception as exc:
        print(f"\n  FAILED ({difficulty}): {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Batch mode ──────────────────────────────────────────────────────────────

def _print_difficulty_summary(
    difficulty: str, results: List[Tuple[str, Optional[Dict]]]
) -> None:
    """Compact comparison table for all memory types tested on one difficulty."""
    level = DIFFICULTY_MAP[difficulty]["level"]
    w = 100
    print(f"\n{'─'*w}")
    print(f"  Summary: {difficulty} (level={level})")
    print(f"{'─'*w}")

    header = (
        f"  {'Memory Type':<24s} {'Success':>8s} {'95% CI':>20s}"
        f" {'AvgSteps':>9s} {'Loop%':>6s} {'Trunc%':>7s}"
    )
    print(header)
    print(f"  {'─'*(w-2)}")

    for mem_type, m in results:
        if m is None:
            print(f"  {mem_type:<24s} {'FAILED':>8s}")
            continue
        sr = float(m.get("success_rate") or 0.0)
        ci_lo = m.get("success_rate_ci95_lower")
        ci_hi = m.get("success_rate_ci95_upper")
        ci_str = (
            f"[{ci_lo:.1%}, {ci_hi:.1%}]"
            if ci_lo is not None and ci_hi is not None else "n/a"
        )
        avg = m.get("avg_steps_all")
        avg_str = f"{avg:.1f}" if avg is not None else "n/a"
        lr = m.get("loop_rate")
        lr_str = f"{float(lr):.0%}" if lr is not None else "n/a"
        tr = m.get("truncation_rate")
        tr_str = f"{float(tr):.0%}" if tr is not None else "n/a"
        print(
            f"  {mem_type:<24s} {sr:>7.1%} {ci_str:>20s}"
            f" {avg_str:>9s} {lr_str:>6s} {tr_str:>7s}"
        )

    print(f"{'─'*w}\n")


def _run_batch(
    base_cfg: DictConfig,
    difficulties: List[str],
    num_episodes: Optional[int],
    max_steps: Optional[int],
    out_base: Optional[str],
) -> None:
    """Run all memory types x difficulties sequentially with per-difficulty summaries.

    Outer loop: difficulties.  Inner loop: memory types.
    Comment out entries in memory_types to skip them — same pattern as evaluator.py.
    """
    # ── Comment / uncomment memory types to include in the batch ──
    memory_types = [
        "baseline",
        "trajectory",
        "reflection",
        "enriched",
        "enriched_history",
        "fade_enriched_history",
    ]

    os.makedirs(out_base or "runs", exist_ok=True)

    total = len(difficulties) * len(memory_types)
    completed = 0
    run_idx = 0
    failed: List[Tuple[str, str, str]] = []

    cfg_episodes = int(getattr(base_cfg.eval, "num_episodes", 0)) or None
    cfg_max_steps = int(getattr(base_cfg.eval, "max_steps_per_episode", 0)) or None
    print(f"Batch: {total} run(s)  ({len(difficulties)} difficulties x {len(memory_types)} memory types)")
    print(f"  Seed: {base_cfg.eval.seed}")
    for d in difficulties:
        s = DIFFICULTY_MAP[d]
        ne = num_episodes or cfg_episodes or s["default_episodes"]
        ms = max_steps or cfg_max_steps or s["default_max_steps"]
        print(f"  {d:<6s}: level={s['level']}, episodes={ne}, max_steps={ms}")

    batch_t0 = time.perf_counter()
    run_durations: List[float] = []

    for diff_idx, diff in enumerate(difficulties, 1):
        level = DIFFICULTY_MAP[diff]["level"]
        print(f"\n{'='*64}")
        print(f"  Difficulty {diff_idx}/{len(difficulties)}: {diff} (level={level})")
        print(f"{'='*64}")

        diff_results: List[Tuple[str, Optional[Dict]]] = []

        for memory_type in memory_types:
            run_idx += 1
            print(f"\n  [{run_idx}/{total}] {memory_type}")

            overrides = OmegaConf.create({
                "agent": {"params": {"memory_type": memory_type}},
            })
            cfg = OmegaConf.merge(base_cfg, overrides)

            run_t0 = time.perf_counter()
            try:
                path, file_metrics = _run_difficulty(
                    difficulty=diff,
                    cfg=cfg,
                    num_episodes=num_episodes,
                    max_steps=max_steps,
                    out_base=out_base,
                    quiet=True,
                )
                if file_metrics is not None:
                    sr = float(file_metrics.get("success_rate") or 0)
                    n = int(file_metrics.get("episodes") or 0)
                    s = int(file_metrics.get("successes") or 0)
                    fname = path.name if path else "n/a"
                    run_secs = time.perf_counter() - run_t0
                    print(
                        f"    -> {sr:.1%} ({s}/{n})  |  {fname}  "
                        f"|  this run {_format_duration(run_secs)}"
                    )
                    diff_results.append((memory_type, file_metrics))
                    completed += 1
                else:
                    diff_results.append((memory_type, None))
                    run_secs = time.perf_counter() - run_t0
                    print(f"    -> FAILED  |  this run {_format_duration(run_secs)}")
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                failed.append((diff, memory_type, msg))
                diff_results.append((memory_type, None))
                run_secs = time.perf_counter() - run_t0
                print(f"    -> FAILED: {msg}  |  this run {_format_duration(run_secs)}")
            finally:
                run_durations.append(time.perf_counter() - run_t0)

            _print_batch_progress_line(batch_t0, run_durations, run_idx, total)

        _print_difficulty_summary(diff, diff_results)

    print(f"\n{'='*64}")
    print(
        f"Batch complete: {completed}/{total} succeeded, {len(failed)} failed  "
        f"|  total wall time {_format_duration(time.perf_counter() - batch_t0)}"
    )
    if failed:
        print("Failed runs:")
        for diff, mem, err in failed:
            print(f"  - {diff} | {mem}: {err}")


# ── Main ────────────────────────────────────────────────────────────────────

def _run_single(
    cfg: DictConfig,
    difficulties: List[str],
    num_episodes: Optional[int],
    max_steps: Optional[int],
    out_base: Optional[str],
) -> None:
    """Default entry: run difficulties sequentially with full per-run output."""
    cfg_episodes = int(getattr(cfg.eval, "num_episodes", 0)) or None
    cfg_max_steps = int(getattr(cfg.eval, "max_steps_per_episode", 0)) or None
    print("TextWorld TreasureHunter Evaluation")
    for d in difficulties:
        s = DIFFICULTY_MAP[d]
        ne = num_episodes or cfg_episodes or s["default_episodes"]
        ms = max_steps or cfg_max_steps or s["default_max_steps"]
        print(f"  {d:<6s}: level={s['level']}, episodes={ne}, max_steps={ms}")

    t0 = time.perf_counter()
    all_results: List[Tuple[str, Optional[Path], Optional[Dict]]] = []

    for diff in difficulties:
        try:
            path, metrics = _run_difficulty(
                difficulty=diff,
                cfg=cfg,
                num_episodes=num_episodes,
                max_steps=max_steps,
                out_base=out_base,
            )
            all_results.append((diff, path, metrics))
        except Exception as exc:
            print(f"ERROR running {diff}: {exc}", file=sys.stderr)
            all_results.append((diff, None, None))

    # Comparison table when running multiple difficulties
    if len(all_results) > 1:
        w = 90
        print(f"\n{'='*w}")
        print("  TextWorld TreasureHunter — Summary")
        print(f"{'='*w}")
        header = (
            f"  {'Difficulty':<10s} {'Level':>5s} {'Success':>8s}"
            f" {'95% CI':>20s} {'AvgSteps':>9s} {'Loop%':>6s} {'Trunc%':>7s}"
        )
        print(header)
        print(f"  {'─'*(w-2)}")
        for diff, _, m in all_results:
            lvl = DIFFICULTY_MAP[diff]["level"]
            if m is None:
                print(f"  {diff:<10s} {lvl:>5d} {'FAILED':>8s}")
                continue
            sr = float(m.get("success_rate") or 0)
            ci_lo = m.get("success_rate_ci95_lower")
            ci_hi = m.get("success_rate_ci95_upper")
            ci_str = (
                f"[{ci_lo:.1%}, {ci_hi:.1%}]"
                if ci_lo is not None and ci_hi is not None else "n/a"
            )
            avg = m.get("avg_steps_all")
            avg_str = f"{avg:.1f}" if avg is not None else "n/a"
            lr_str = f"{float(m.get('loop_rate', 0)):.0%}"
            tr_str = f"{float(m.get('truncation_rate', 0)):.0%}"
            print(
                f"  {diff:<10s} {lvl:>5d} {sr:>7.1%} {ci_str:>20s}"
                f" {avg_str:>9s} {lr_str:>6s} {tr_str:>7s}"
            )
        print(f"{'='*w}")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal elapsed: {_format_duration(elapsed)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run adaptive-memento agent on TextWorld TreasureHunter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluator_textworld.py --difficulty easy --num-episodes 5\n"
            "  python evaluator_textworld.py --difficulty all\n"
            "  python evaluator_textworld.py --difficulty hard --max-steps 200\n"
            "  python evaluator_textworld.py --batch\n"
            "  python evaluator_textworld.py --batch --difficulty easy\n"
        ),
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run all difficulty x memory type combinations (full experiment grid)",
    )
    parser.add_argument(
        "--config", default="config/config_textworld.yaml",
        help="Path to config YAML (default: config/config_textworld.yaml)",
    )
    parser.add_argument(
        "--difficulty", default="all",
        choices=["easy", "medium", "hard", "all"],
        help="TreasureHunter difficulty (default: all)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=None,
        help="Override number of episodes per difficulty",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override max steps per episode",
    )
    parser.add_argument(
        "--out-json", default=None,
        help="Base output directory for JSONL/JSON (default: runs/)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: Config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = OmegaConf.load(str(cfg_path))

    difficulties = (
        ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]
    )

    if args.batch:
        _run_batch(cfg, difficulties, args.num_episodes, args.max_steps, args.out_json)
    else:
        _run_single(cfg, difficulties, args.num_episodes, args.max_steps, args.out_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
