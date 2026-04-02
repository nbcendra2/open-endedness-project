"""Generate TextWorld Coin Collector game files (.z8) on demand and cache them.

The generator first tries the TextWorld Python API, then falls back to the
``tw-make`` CLI tool (which is confirmed to work on this install).
"""

import logging
import os
import subprocess
import threading

logger = logging.getLogger(__name__)

GAMES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "games",
)

# Serialize all game generation so concurrent threads never write the same
# file (or its intermediate .json / .ni artefacts) simultaneously.
_generation_lock = threading.Lock()


def _generate_via_python_api(level: int, seed: int, filepath: str) -> str:
    """Attempt game generation through the TextWorld Python API."""
    import textworld
    import textworld.generator
    from textworld.challenges.coin_collector import make as make_coin_game

    options = textworld.GameOptions()
    options.path = filepath
    options.seeds = seed

    game = make_coin_game(settings={"level": level}, options=options)
    game_file = textworld.generator.compile_game(game, options)
    # compile_game returns the filepath of the compiled game
    return str(game_file)


def _generate_via_cli(level: int, seed: int, filepath: str) -> str:
    """Fallback: shell out to ``tw-make`` which is known to work."""
    # Try with --seed first (not all versions support it)
    for cmd in [
        [
            "tw-make", "tw-coin_collector",
            "--level", str(level),
            "--seed", str(seed),
            "--output", filepath,
        ],
        [
            "tw-make", "tw-coin_collector",
            "--level", str(level),
            "--output", filepath,
        ],
    ]:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(filepath):
            return filepath

    raise RuntimeError(
        f"tw-make failed to generate {filepath}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


class CoinCollectorGameGenerator:
    """Generate and cache TextWorld Coin Collector ``.z8`` game files.

    Each ``(level, seed)`` pair maps to a unique file under ``games/``.
    Files are generated once and reused across providers / memory types.

    Thread safety: generation is serialized via a module-level lock so
    concurrent workers never write the same file simultaneously.
    """

    def __init__(self, level: int = 1):
        self.level = level
        os.makedirs(GAMES_DIR, exist_ok=True)

    def get_game_file(self, seed: int = 0) -> str:
        """Return the path to a ``.z8`` game file, generating it if needed.

        Uses double-checked locking: the fast path (file already cached)
        needs no lock; only actual generation is serialized.

        Always returns a **string filepath** (never a Game object).
        """
        filename = f"tw_coin_L{self.level}_s{seed}.z8"
        filepath = os.path.join(GAMES_DIR, filename)

        # Fast path: file already cached — no lock needed
        if os.path.exists(filepath):
            logger.debug("Reusing cached game: %s", filepath)
            return filepath

        # Slow path: serialize generation across all threads
        with _generation_lock:
            # Double-check: another thread may have generated it while we waited
            if os.path.exists(filepath):
                logger.debug("Reusing cached game (after lock): %s", filepath)
                return filepath

            # Python API first, CLI fallback
            try:
                game_file = _generate_via_python_api(self.level, seed, filepath)
                logger.info("Generated game (Python API): %s", game_file)
                # Ensure we got a filepath string back
                if os.path.exists(str(game_file)):
                    return str(game_file)
            except Exception as exc:
                logger.debug("Python API generation failed (%s); trying CLI …", exc)

            game_file = _generate_via_cli(self.level, seed, filepath)
            logger.info("Generated game (CLI): %s", game_file)
            return game_file

    def ensure_games_exist(self, seeds: list[int]) -> None:
        """Pre-generate all game files for the given seeds.

        Call this from the main thread before spawning workers so the
        fast path in ``get_game_file`` always hits for every seed.
        """
        for seed in seeds:
            self.get_game_file(seed)