"""Run the full TextWorld Coin Collector experiment grid.

Executes all (provider × memory type × difficulty level) combinations
with provider-level parallelism (each provider runs on its own thread).

Usage:
    # Full grid — providers run in parallel
    python run_tw_batch.py

    # Subset, 4 workers
    python run_tw_batch.py --levels L1 --memory-types baseline,trajectory --providers openai --workers 4

    # More episodes for tighter CIs
    python run_tw_batch.py --episodes 20

    # Custom output directory
    python run_tw_batch.py --output-dir results/experiment_01
"""

import argparse
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omegaconf import OmegaConf

from evaluator import (
    Evaluator,
    _emit_rollout_summary,
    _format_duration,
    _print_env_summary,
)

# Maps provider key -> default model name
PROVIDER_MODELS = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.5-flash-lite",
}

# Thread-safe console output
_print_lock = threading.Lock()


def _ts_print(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Pre-generate game files (run on main thread before workers start)
# ---------------------------------------------------------------------------

def _pregeneratate_game_files(
    levels: list[str],
    num_episodes: int,
    base_seed: int,
) -> None:
    """Sequentially generate all .z8 files so worker threads only hit cache.

    This eliminates the race condition where multiple threads try to
    generate the same game file simultaneously.
    """
    from textworld_env.tw_coin_collector import CoinCollectorGameGenerator

    seeds = [base_seed + i for i in range(num_episodes)]
    for level_str in levels:
        parsed_level = int(level_str.upper().replace("L", ""))
        generator = CoinCollectorGameGenerator(level=parsed_level)
        print(f"  Pre-generating {len(seeds)} game file(s) for level {level_str} …")
        generator.ensure_games_exist(seeds)
    print(f"  All game files ready.\n")


# ---------------------------------------------------------------------------
# Single run (executed inside a worker thread)
# ---------------------------------------------------------------------------

def _run_single(
    base_cfg,
    config_yaml: str,
    level: str,
    provider: str,
    model: str,
    memory_type: str,
    output_dir: Path,
    run_label: str,
) -> dict:
    """Execute one (level, provider, memory_type) combination.

    Returns a dict with keys: level, provider, memory_type, success,
    metrics | error, duration.
    """
    env = f"tw-coin_collector-{level}"
    out_jsonl = str(output_dir / f"tw_{provider}_{level}_{memory_type}.jsonl")

    overrides = OmegaConf.create({
        "env": {"name": env, "fallback_action": "look"},
        "agent": {"params": {"memory_type": memory_type}},
        "llm": {"provider": provider, "name": model},
        "eval": {"out_json": out_jsonl},
    })
    cfg = OmegaConf.merge(base_cfg, overrides)

    t0 = time.perf_counter()
    try:
        evaluator = Evaluator(cfg)
        result = evaluator.run()
        evaluator.close()

        rollout_path = Path(result["out_path"])
        file_metrics = _emit_rollout_summary(rollout_path, config_yaml)
        sr = float(file_metrics.get("success_rate") or 0)
        n = int(file_metrics.get("episodes") or 0)
        s = int(file_metrics.get("successes") or 0)
        duration = time.perf_counter() - t0

        _ts_print(
            f"  ✓ {run_label}  ->  {sr:.1%} ({s}/{n})  |  "
            f"{rollout_path.name}  |  {_format_duration(duration)}"
        )
        return {
            "level": level,
            "provider": provider,
            "memory_type": memory_type,
            "success": True,
            "metrics": file_metrics,
            "duration": duration,
        }
    except Exception as exc:
        duration = time.perf_counter() - t0
        msg = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        _ts_print(f"  ✗ {run_label}  ->  FAILED: {msg}  |  {_format_duration(duration)}\n{tb}")
        return {
            "level": level,
            "provider": provider,
            "memory_type": memory_type,
            "success": False,
            "error": msg,
            "duration": duration,
        }


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------

def _run_tw_batch(
    base_cfg,
    config_yaml: str = "config/coin_collector.yaml",
    levels: list[str] | None = None,
    memory_types: list[str] | None = None,
    providers: list[str] | None = None,
    output_dir: str = "results",
    max_workers: int | None = None,
    num_episodes: int | None = None,
):
    """Run the experiment grid with provider-level parallelism."""
    if levels is None:
        levels = ["L1", "L5", "L10"]
    if memory_types is None:
        memory_types = [
            "baseline",
            "trajectory",
            "reflection",
            "enriched",
            "enriched_history",
            "fade_enriched_history",
        ]
    if providers is None:
        providers = list(PROVIDER_MODELS.keys())

    for p in providers:
        if p not in PROVIDER_MODELS:
            raise ValueError(
                f"Unknown provider '{p}'. Choose from: {', '.join(PROVIDER_MODELS)}"
            )

    # Override episode count if requested
    if num_episodes is not None:
        base_cfg = OmegaConf.merge(
            base_cfg,
            OmegaConf.create({"eval": {"num_episodes": num_episodes}}),
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default: one worker per provider (parallel across APIs)
    if max_workers is None:
        max_workers = min(len(providers), 3)

    # Build the full job list
    jobs: list[dict] = []
    for level in levels:
        for provider in providers:
            for memory_type in memory_types:
                jobs.append({
                    "level": level,
                    "provider": provider,
                    "model": PROVIDER_MODELS[provider],
                    "memory_type": memory_type,
                })

    total = len(jobs)
    eps = int(base_cfg.eval.num_episodes)
    seed = int(base_cfg.eval.seed)

    print(f"TextWorld Batch: {total} run(s)  |  {max_workers} worker(s)")
    print(f"  Levels           : {', '.join(levels)}")
    print(f"  Providers        : {', '.join(f'{p} ({PROVIDER_MODELS[p]})' for p in providers)}")
    print(f"  Memory types     : {', '.join(memory_types)}")
    print(f"  Episodes per run : {eps}")
    print(f"  Max steps/episode: {base_cfg.eval.max_steps_per_episode}")
    print(f"  Seed             : {seed}")
    print(f"  Output directory : {output_path.resolve()}")
    print()

    # ------------------------------------------------------------------
    # Pre-generate all game files on the main thread so worker threads
    # never race on file creation.
    # ------------------------------------------------------------------
    _pregeneratate_game_files(levels, eps, seed)

    batch_t0 = time.perf_counter()
    results: list[dict] = []
    completed = 0
    failed_runs: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {}
        for idx, job in enumerate(jobs, 1):
            label = (
                f"[{idx}/{total}] CC-{job['level']} / "
                f"{job['provider']} / {job['memory_type']}"
            )
            future = pool.submit(
                _run_single,
                base_cfg,
                config_yaml,
                job["level"],
                job["provider"],
                job["model"],
                job["memory_type"],
                output_path,
                label,
            )
            future_to_job[future] = job

        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)
            if result["success"]:
                completed += 1
            else:
                failed_runs.append(result)

            done = len(results)
            elapsed = time.perf_counter() - batch_t0
            avg = elapsed / done
            eta = avg * (total - done)
            _ts_print(
                f"  Progress: {done}/{total} done  |  "
                f"elapsed {_format_duration(elapsed)}  |  "
                f"ETA {_format_duration(eta)}"
            )

    # ---------------------------------------------------------------------------
    # Summary tables: group by (level, provider)
    # ---------------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("SUMMARY TABLES")
    print(f"{'='*80}")

    for level in levels:
        for provider in providers:
            model = PROVIDER_MODELS[provider]
            env = f"tw-coin_collector-{level}"
            env_results: list[tuple[str, dict | None]] = []

            for memory_type in memory_types:
                match = [
                    r for r in results
                    if r["level"] == level
                    and r["provider"] == provider
                    and r["memory_type"] == memory_type
                ]
                if match and match[0]["success"]:
                    env_results.append((memory_type, match[0]["metrics"]))
                else:
                    env_results.append((memory_type, None))

            summary_label = f"{env} ({provider}/{model})"
            _print_env_summary(summary_label, env_results)

    # ---------------------------------------------------------------------------
    # Final status
    # ---------------------------------------------------------------------------
    wall = _format_duration(time.perf_counter() - batch_t0)
    print(f"\n{'='*80}")
    print(
        f"Batch complete: {completed}/{total} succeeded, "
        f"{len(failed_runs)} failed  |  total wall time {wall}"
    )
    if failed_runs:
        print("Failed runs:")
        for r in failed_runs:
            print(
                f"  - CC-{r['level']} | {r['provider']} | "
                f"{r['memory_type']}: {r['error']}"
            )
            
    # --- Auto-generate CSV ---
    from collect_results_csv import collect_summaries, write_csv
    csv_rows = collect_summaries(output_path)
    if csv_rows:
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = write_csv(csv_rows, output_path / f"all_results_{ts}.csv")
        _ts_print(f"\nWrote CSV: {csv_path}  ({len(csv_rows)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TextWorld Coin Collector experiment grid"
    )
    parser.add_argument(
        "--config", default="config/coin_collector.yaml",
        help="Base config YAML",
    )
    parser.add_argument(
        "--levels", default=None,
        help="Comma-separated levels, e.g. L1,L5,L10  (default: L1,L5,L10)",
    )
    parser.add_argument(
        "--memory-types", default=None,
        help="Comma-separated memory types, e.g. baseline,trajectory",
    )
    parser.add_argument(
        "--providers", default=None,
        help="Comma-separated LLM providers, e.g. openai,deepseek,gemini  "
             "(default: all three)",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for result files (default: results/)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers (default: one per provider)",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override number of episodes per run (default: from config)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    levels = args.levels.split(",") if args.levels else None
    mem_types = args.memory_types.split(",") if args.memory_types else None
    provs = args.providers.split(",") if args.providers else None

    _run_tw_batch(
        cfg,
        args.config,
        levels=levels,
        memory_types=mem_types,
        providers=provs,
        output_dir=args.output_dir,
        max_workers=args.workers,
        num_episodes=args.episodes,
    )