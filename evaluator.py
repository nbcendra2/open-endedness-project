"""Functionality: Run the agent on episodes and write per-episode results to JSONL

What this file does in one sentence: it wires config to env plus agent, runs episodes,
and logs each episode as one JSON line so you can analyze or postprocess later

Single run (default): load a YAML config, build env and agent, play num_episodes,
write a timestamped JSONL under runs (or whatever eval.out_json points at), then
compute the same metrics as postprocess_rollouts (success rate, step stats), print them,
and write a sibling *_summary.json next to the rollout

Batch run (--batch): repeat many experiments by merging overrides into the base YAML
in memory only, so config on disk never changes between runs. After each rollout,
builds a small summary JSON with the same metrics logic as postprocess_rollouts.py

Why config_snapshot on episode 0: so each rollout file carries the exact config used
for that run. Postprocessing can trust the file instead of re-reading a YAML that may
have changed since the experiment
"""

from environments import make_env

from typing import Any, Dict, List
import itertools
import json
import os
import time
from omegaconf import OmegaConf, DictConfig

from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import argparse

from agent import build_agent
from postprocess_rollouts import episode_has_loop, is_episode_success


def _format_duration(seconds: float) -> str:
    """Short human-readable duration (e.g. 90.3 -> '1m 30s')."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _batch_eta_seconds(finished_durations: list[float], remaining_runs: int) -> float | None:
    if remaining_runs <= 0 or not finished_durations:
        return None
    return _mean(finished_durations) * remaining_runs


def _print_batch_progress_line(
    batch_t0: float, run_durations: list[float], run_idx: int, total: int
) -> None:
    """After each sub-run: wall-clock elapsed + ETA from mean duration × remaining."""
    elapsed_batch = time.perf_counter() - batch_t0
    remaining = total - run_idx
    eta_s = _batch_eta_seconds(run_durations, remaining)
    parts = [f"elapsed {_format_duration(elapsed_batch)}"]
    if eta_s is not None:
        finish = datetime.now() + timedelta(seconds=eta_s)
        parts.append(f"~{_format_duration(eta_s)} left (est.)")
        parts.append(f"ETA ~{finish.strftime('%H:%M:%S')}")
    print(f"    Batch: {'  |  '.join(parts)}")


def _emit_rollout_summary(rollout_path: Path, config_yaml: str) -> dict[str, Any]:
    """Re-read JSONL, compute metrics (same as postprocess_rollouts), write *_summary.json

    Returns the per-file metrics dict for console reporting
    """
    from postprocess_rollouts import (
        _embedded_config_from_rows,
        _file_metrics_from_rows,
        _load_jsonl,
        _maybe_load_config,
    )

    rows = _load_jsonl(rollout_path)
    file_metrics = _file_metrics_from_rows(rows, rollout_path)
    embedded = _embedded_config_from_rows(rows)
    config_snapshot = embedded if embedded is not None else _maybe_load_config(config_yaml)
    summary = {
        "input_file": str(rollout_path),
        "metrics": file_metrics,
        "config_snapshot": config_snapshot,
        "config_snapshot_source": "rollout_jsonl" if embedded is not None else "yaml_file",
    }
    summary_path = rollout_path.with_name(f"{rollout_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_metrics


def _print_run_metrics(m: dict[str, Any]) -> None:
    """Short human-readable line(s) after a multi-episode run"""
    n = int(m.get("episodes") or 0)
    s = int(m.get("successes") or 0)
    sr = float(m.get("success_rate") or 0.0)
    ci_lo = m.get("success_rate_ci95_lower")
    ci_hi = m.get("success_rate_ci95_upper")
    ci_str = ""
    if ci_lo is not None and ci_hi is not None:
        ci_str = f"  95% CI [{ci_lo:.1%}, {ci_hi:.1%}]"
    print(f"Metrics: success_rate={sr:.1%} ({s}/{n} episodes){ci_str}")
    lr = m.get("loop_rate")
    if lr is not None and n > 0:
        lw = int(m.get("episodes_with_loop") or 0)
        print(f"         loop_rate={float(lr):.1%} ({lw}/{n} episodes)")
    tr = m.get("truncation_rate")
    if tr is not None and n > 0:
        et = int(m.get("episodes_truncated") or 0)
        print(f"         truncation_rate={float(tr):.1%} ({et}/{n} episodes)")
    bfr = m.get("blocked_forward_rate")
    fa = int(m.get("forward_attempt_steps") or 0)
    if fa > 0 and bfr is not None:
        bs = int(m.get("blocked_forward_steps") or 0)
        print(f"         blocked_forward_rate={float(bfr):.1%} ({bs}/{fa} go-forward steps)")
    elif fa == 0:
        print("         blocked_forward_rate=n/a (no go-forward steps logged)")
    avg_all = m.get("avg_steps_all")
    if avg_all is not None:
        print(f"         avg_steps_all={avg_all:.2f} (std={m.get('std_steps_all')})")
    avg_ok = m.get("avg_steps_to_success")
    if avg_ok is not None:
        print(
            f"         avg_steps_to_success={avg_ok:.2f} "
            f"(std={m.get('std_steps_to_success')})"
        )
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


class Evaluator:
    """Runs BabyAI-style episodes: reset env, loop act or step, append step logs

    One Evaluator instance is tied to one resolved config. It owns the gym env and
    the agent. Call run to play all episodes and write JSONL, then close to free the env
    """

    def __init__(self, config):
        """Build env and agent from OmegaConf, read eval limits, freeze config for logging

        config is expected to expose env.name, env.invalid_action_mode, env.fallback_action,
        optional env.gym_kwargs, eval.seed, eval.num_episodes, eval.max_steps_per_episode,
        eval.out_json, and agent.params.memory_type (defaults to baseline if missing)

        _config_snapshot is a plain dict copy of the full tree after resolve. It is written
        only on the first JSON line of the rollout so downstream tools know what actually ran
        """
        self.config = config
        self.env_name = self.config.env.name
        # Gym kwargs are optional; empty dict if absent so make_env always gets a dict
        env_gym_kwargs = (
            OmegaConf.to_container(self.config.env.gym_kwargs, resolve=True)
            if "gym_kwargs" in self.config.env and self.config.env.gym_kwargs is not None
            else {}
        )

        self.env = make_env(
            env_name=self.env_name,
            gym_kwargs=env_gym_kwargs,
            invalid_action_mode=self.config.env.invalid_action_mode,
            fallback_action=self.config.env.fallback_action,
        )
        self.seed = int(self.config.eval.seed)
        self.num_episodes = int(self.config.eval.num_episodes)
        self.max_steps = int(self.config.eval.max_steps_per_episode)
        self.out_json = self.config.eval.out_json

        # system_prompt is None here; the real instruction comes from the env after reset
        self.agent = build_agent(config=self.config, system_prompt=None)
        self.memory_type = str(
            getattr(getattr(self.config.agent, "params", {}), "memory_type", "baseline")
        ).lower()
        # agent_type labels the output filename and JSON lines (same as memory flavor)
        self.agent_type = self.memory_type
        # Frozen copy of the full config used for this run (stored on first JSONL line only)
        self._config_snapshot = OmegaConf.to_container(self.config, resolve=True)

    def run_episode(self, episode_idx):
        """Play one episode from reset until done or max_steps, return episode dict

        Flow: reset with a deterministic offset seed, tell the agent the mission,
        then for each timestep call agent.act, env.step, agent.observe_step. The list
        steps holds one record per env step for debugging and analysis

        Returns keys: episode, seed, num_steps, total_reward, terminated, truncated, success, has_loop, steps
        """
        state = self.env.reset(seed=self.seed + episode_idx)
        system_prompt = self.env.get_instruction_prompt(state.mission)

        self.agent.start_episode(
            episode_id=episode_idx, mission=state.mission,
            seed=self.seed + episode_idx, system_prompt=system_prompt,
            env_name=self.env_name,
        )

        steps: List[Dict[str, Any]] = []
        total_reward = 0.0
        terminated = False
        truncated = False

        for t in range(self.max_steps):
            # LLM chooses an action name from valid_actions; reason is for logging only
            out = self.agent.act(
                text_obs=state.text_obs,
                mission=state.mission,
                valid_actions=state.valid_actions,
                step_idx=t,
            )

            proposed_action = str(out.get("action", ""))

            pos_before = self.env.get_agent_grid_position()
            # Env may remap invalid actions depending on invalid_action_mode
            step = self.env.step(proposed_action)
            pos_after = self.env.get_agent_grid_position()

            # Memory agents update trajectory or reflection buffers here
            self.agent.observe_step(step_idx=t, prev_text_obs=state.text_obs,
                    action=proposed_action, step_result=step)
            total_reward += step.reward

            au = str(step.action_used).strip().lower()
            movement_attempt = au == "go forward"
            movement_blocked = (
                movement_attempt
                and pos_before is not None
                and pos_after is not None
                and pos_before == pos_after
            )

            steps.append(
                {
                    "episode": episode_idx,
                    "step_idx": t,
                    "mission": state.mission,
                    "text_obs": state.text_obs,
                    "valid_actions": state.valid_actions,
                    "agent_output": {
                        "action": proposed_action,
                        "reason": out.get("reason", ""),
                    },
                    "action_used": step.action_used,
                    "action_was_valid": step.action_was_valid,
                    "reward": step.reward,
                    "terminated": step.terminated,
                    "truncated": step.truncated,
                    "env_reason": step.reason,
                    "transition": {
                        "movement_attempt": movement_attempt,
                        "movement_blocked": movement_blocked,
                    },
                }
            )
            state = step.state
            terminated = step.terminated
            truncated = step.truncated
            if terminated or truncated:
                break
        # Lets the agent run end-of-episode reflection or memory flush if implemented
        self.agent.end_episode(total_reward=total_reward, terminated=terminated)

        result = {
            "episode": episode_idx,
            "seed": self.seed + episode_idx,
            "num_steps": len(steps),
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "steps": steps,
        }
        # Fade-enriched memory diagnostics (only for fade_enriched_history)
        if self.memory_type == "fade_enriched_history":
            buf = getattr(self.agent, "_enriched_buffer", [])
            result["memory_stats"] = {
                "active_facts": sum(1 for f in buf if f.get("state") == "active"),
                "dormant_facts": sum(1 for f in buf if f.get("state") == "dormant"),
                "fused_facts": sum(1 for f in buf if f.get("fused_from", 0) > 0),
                "reactivations": self.agent._fade_reactivation_count,
                "priming_boosts": self.agent._fade_priming_count,
                "tag_triggers_fired": self.agent._fade_tag_trigger_count,
                "tag_shields_expired": self.agent._fade_tag_expire_count,
                "contradictions": self.agent._fade_contradiction_count,
                "compatible_resolved": self.agent._fade_compatible_count,
                "subsume_resolved": self.agent._fade_subsume_count,
                "fusions_performed": self.agent._fade_fusion_count,
            }
        # Aligns with postprocess_rollouts (SR / s_i, loop l_i)
        result["success"] = is_episode_success(result)
        result["has_loop"] = episode_has_loop(result)
        return result

    def run(self):
        """Loop run_episode for all episodes, append one JSON object per line to a file

        Output path: stem from eval.out_json, forced to jsonl suffix, then agent_type and
        timestamp inserted so repeated runs do not overwrite each other

        First line only includes config_snapshot so tools can recover the run config without
        guessing from the current YAML on disk

        Returns a small dict with paths and limits for callers (batch mode uses out_path)
        """
        out_dir = os.path.dirname(self.out_json)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        p = Path(self.out_json)
        if p.suffix.lower() != ".jsonl":
            p = p.with_suffix(".jsonl")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_tag = self.env_name.replace("/", "_").replace("-", "")
        out_path = p.with_name(f"{p.stem}_{self.agent_type}_{env_tag}_{ts}{p.suffix}")

        with open(out_path, "w", encoding="utf-8") as f:
            for ep in tqdm(range(self.num_episodes), total=self.num_episodes, desc="Episodes"):
                episode_result = self.run_episode(ep)
                line = {
                    "env_name": self.env_name,
                    "agent_type": self.agent_type,
                    "run_seed": self.seed,
                    "max_steps": self.max_steps,
                    "num_episodes": self.num_episodes,
                    **episode_result,
                }
                # Episode 0 only: embed full config so rollouts stay self-describing after config.yaml edits
                if ep == 0:
                    line["config_snapshot"] = self._config_snapshot
                json.dump(line, f, ensure_ascii=False)
                f.write("\n")

        return {
            "env_name": self.env_name,
            "num_episodes": self.num_episodes,
            "max_steps": self.max_steps,
            "out_path": str(out_path),
            "format": "jsonl",
        }

    def close(self) -> None:
        """Release underlying gym resources"""
        self.env.close()


def _run_single(cfg: DictConfig, config_yaml: str = "config/config.yaml") -> None:
    """Default entry: one Evaluator, one rollout file, metrics summary on disk and stdout"""
    t0 = time.perf_counter()
    evaluator = Evaluator(cfg)
    result = evaluator.run()
    evaluator.close()
    out_path = Path(result["out_path"])
    print(f"Wrote rollout: {out_path}")
    metrics = _emit_rollout_summary(out_path, config_yaml)
    summary_path = out_path.with_name(f"{out_path.stem}_summary.json")
    print(f"Wrote summary: {summary_path}")
    print(f"Elapsed (rollout + summary): {_format_duration(time.perf_counter() - t0)}")
    _print_run_metrics(metrics)


def _print_env_summary(env: str, results: list[tuple[str, dict | None]]) -> None:
    """Compact comparison table for all memory types tested on one environment."""
    label = env.replace("BabyAI-", "").replace("-v0", "")
    w = 100
    print(f"\n{'─'*w}")
    print(f"  Summary: {label}")
    print(f"{'─'*w}")

    header = (
        f"  {'Memory Type':<24s} {'Success':>8s} {'95% CI':>20s}"
        f" {'AvgSteps':>9s} {'Loop%':>6s} {'Trunc%':>7s} {'Blocked%':>9s}"
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
            if ci_lo is not None and ci_hi is not None
            else "n/a"
        )
        avg = m.get("avg_steps_all")
        avg_str = f"{avg:.1f}" if avg is not None else "n/a"
        lr = m.get("loop_rate")
        lr_str = f"{float(lr):.0%}" if lr is not None else "n/a"
        tr = m.get("truncation_rate")
        tr_str = f"{float(tr):.0%}" if tr is not None else "n/a"
        bfr = m.get("blocked_forward_rate")
        bfr_str = f"{float(bfr):.0%}" if bfr is not None else "n/a"
        print(
            f"  {mem_type:<24s} {sr:>7.1%} {ci_str:>20s}"
            f" {avg_str:>9s} {lr_str:>6s} {tr_str:>7s} {bfr_str:>9s}"
        )

    print(f"{'─'*w}\n")


def _run_batch(base_cfg: DictConfig, config_yaml: str = "config/config.yaml") -> None:
    """Run all memory types x environments sequentially with per-env summaries.

    Iterates environments in the outer loop and memory types in the inner loop.
    After finishing all memory types for one environment a compact comparison
    table is printed.  Crashes are caught per-run so the batch continues.
    """
    envs = [
        "BabyAI-GoToObj-v0",
        "BabyAI-Open-v0",
        "BabyAI-Unlock-v0",
        "BabyAI-Pickup-v0",
        # BabyAI-PutNext-v0 is not registered in current Minigrid; S5N2 matches common docs
        "BabyAI-PutNextS5N2-v0",
    ]
    # Phase 1: five lighter memory modes. Run fade_enriched_history separately later
    # with the same config (seed, episodes, max_steps) for comparability.
    memory_types = [
        "baseline",
        "trajectory",
        "reflection",
        "enriched",
        "enriched_history",
    ]

    os.makedirs("runs", exist_ok=True)

    total = len(envs) * len(memory_types)
    completed = 0
    run_idx = 0
    failed: list[tuple[str, str, str]] = []

    print(f"Batch: {total} run(s)  ({len(envs)} envs × {len(memory_types)} memory types)")
    print(f"  Episodes per run : {base_cfg.eval.num_episodes}")
    print(f"  Max steps/episode: {base_cfg.eval.max_steps_per_episode}")
    print(f"  Seed             : {base_cfg.eval.seed}")

    batch_t0 = time.perf_counter()
    run_durations: list[float] = []

    for env_idx, env in enumerate(envs, 1):
        env_label = env.replace("BabyAI-", "").replace("-v0", "")
        print(f"\n{'='*64}")
        print(f"  Environment {env_idx}/{len(envs)}: {env_label}")
        print(f"{'='*64}")

        env_results: list[tuple[str, dict | None]] = []

        for memory_type in memory_types:
            run_idx += 1
            print(f"\n  [{run_idx}/{total}] {memory_type}")

            overrides = OmegaConf.create({
                "env": {"name": env},
                "agent": {"params": {"memory_type": memory_type}},
            })
            cfg = OmegaConf.merge(base_cfg, overrides)

            run_t0 = time.perf_counter()
            try:
                evaluator = Evaluator(cfg)
                result = evaluator.run()
                evaluator.close()

                rollout_path = Path(result["out_path"])
                file_metrics = _emit_rollout_summary(rollout_path, config_yaml)
                sr = float(file_metrics.get("success_rate") or 0)
                n = int(file_metrics.get("episodes") or 0)
                s = int(file_metrics.get("successes") or 0)
                run_secs = time.perf_counter() - run_t0
                print(
                    f"    -> {sr:.1%} ({s}/{n})  |  {rollout_path.name}  "
                    f"|  this run {_format_duration(run_secs)}"
                )
                env_results.append((memory_type, file_metrics))
                completed += 1
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                failed.append((env, memory_type, msg))
                env_results.append((memory_type, None))
                run_secs = time.perf_counter() - run_t0
                print(f"    -> FAILED: {msg}  |  this run {_format_duration(run_secs)}")
            finally:
                run_durations.append(time.perf_counter() - run_t0)

            _print_batch_progress_line(batch_t0, run_durations, run_idx, total)

        _print_env_summary(env, env_results)

    print(f"\n{'='*64}")
    print(
        f"Batch complete: {completed}/{total} succeeded, {len(failed)} failed  "
        f"|  total wall time {_format_duration(time.perf_counter() - batch_t0)}"
    )
    if failed:
        print("Failed runs:")
        for env, mem, err in failed:
            print(f"  - {env} | {mem}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument(
        "--batch", action="store_true",
        help="Run all env × memory type combinations (full experiment grid)",
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to the base config YAML file",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.batch:
        _run_batch(cfg, args.config)
    else:
        _run_single(cfg, args.config)
