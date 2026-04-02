"""TextWorld environment wrapper — same EnvState / StepResult interface as EnvWrapper.

The agent and evaluator work identically across BabyAI and TextWorld because
both wrappers expose the same dataclass contract.

Env-name format
---------------
    tw-coin_collector-L{level}

Examples:
    tw-coin_collector-L1   (easy)
    tw-coin_collector-L5   (medium)
    tw-coin_collector-L10  (hard)
"""

import logging
import os
import threading
from typing import Optional

from environments.env_wrapper import EnvState, StepResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TextWorld's Inform7 grammar parser and Jericho interpreter use process-wide
# global state that is NOT thread-safe.  Serialising start / reset / step /
# close behind a single lock is safe because these calls take milliseconds;
# the real latency (LLM API) remains fully parallel across threads.
# ---------------------------------------------------------------------------
_tw_runtime_lock = threading.Lock()

# Fallback actions when admissible_commands is unavailable
_DEFAULT_ACTIONS = [
    "go north", "go south", "go east", "go west",
    "take coin", "look", "inventory",
    "open door",
]


def _parse_tw_env_name(env_name: str) -> dict:
    """Extract parameters from an env name like ``tw-coin_collector-L5``."""
    level = 1
    for part in env_name.split("-"):
        upper = part.upper()
        if upper.startswith("L") and upper[1:].isdigit():
            level = int(upper[1:])
    return {"level": level}


class TextWorldEnvWrapper:
    """Wrap TextWorld Coin Collector with the EnvState / StepResult interface.

    Parameters
    ----------
    env_name : str
        Identifier such as ``tw-coin_collector-L5``.
    invalid_action_mode : str
        ``'fallback'`` silently replaces bad actions; ``'strict'`` raises.
    fallback_action : str
        Action used on fallback (default ``'look'`` — a harmless no-op).
    gym_kwargs : dict | None
        Unused; accepted for call-signature compatibility with ``EnvWrapper``.
    """

    def __init__(
        self,
        env_name: str = "tw-coin_collector-L1",
        invalid_action_mode: str = "fallback",
        fallback_action: str = "look",
        gym_kwargs: dict | None = None,
    ):
        try:
            import textworld
        except ImportError as exc:
            raise ImportError(
                "TextWorld is required.  Install with:  pip install textworld"
            ) from exc
        self._textworld = textworld

        parsed = _parse_tw_env_name(env_name)
        self.level = parsed["level"]
        self.env_name = env_name
        self.invalid_action_mode = invalid_action_mode
        self.fallback_action = fallback_action

        from textworld_env.tw_coin_collector import CoinCollectorGameGenerator
        self._generator = CoinCollectorGameGenerator(level=self.level)

        # TextWorld 1.7 uses request_infos keyword
        self._request_infos = textworld.EnvInfos(
            description=True,
            inventory=True,
            admissible_commands=True,
            won=True,
            lost=True,
        )

        self._env = None
        self._game_state = None
        self._mission: str = ""
        self._current_valid_actions: list[str] = []

    # ---- EnvState / StepResult translation ----

    def _to_state(self, game_state) -> EnvState:
        description = getattr(game_state, "description", None) or ""
        inventory = getattr(game_state, "inventory", None) or ""
        feedback = getattr(game_state, "feedback", None) or ""
        admissible = list(getattr(game_state, "admissible_commands", None) or [])

        # If admissible_commands is empty/None, provide sensible defaults
        if not admissible:
            admissible = list(_DEFAULT_ACTIONS)

        # Compose the text observation the LLM will see
        text_parts: list[str] = []
        if feedback:
            text_parts.append(feedback.strip())
        # Append room description only when feedback doesn't already contain it
        if description and description.strip() not in (feedback or ""):
            text_parts.append(f"\nRoom description: {description.strip()}")
        if inventory and "nothing" not in inventory.lower():
            text_parts.append(f"\nInventory: {inventory.strip()}")

        text_obs = "\n".join(text_parts) if text_parts else "You see nothing special."
        mission = getattr(game_state, "objective", None) or self._mission
        self._current_valid_actions = admissible

        return EnvState(
            mission=mission,
            text_obs=text_obs,
            valid_actions=admissible,
            raw_obs={
                "feedback": feedback,
                "description": description,
                "inventory": inventory,
            },
            raw_info={
                "admissible_commands": admissible,
                "won": getattr(game_state, "won", False),
                "lost": getattr(game_state, "lost", False),
                "score": getattr(game_state, "score", 0),
                "max_score": getattr(game_state, "max_score", 1),
            },
        )

    # ---- public interface (mirrors EnvWrapper) ----

    def reset(self, seed: Optional[int] = None) -> EnvState:
        """Load (and possibly generate) a game for *seed*, then reset it."""
        if self._env is not None:
            try:
                with _tw_runtime_lock:
                    self._env.close()
            except Exception:
                pass

        game_file = self._generator.get_game_file(seed if seed is not None else 0)

        if not isinstance(game_file, str) or not os.path.exists(game_file):
            raise RuntimeError(
                f"Expected a .z8 filepath from get_game_file, got: {game_file!r}"
            )

        with _tw_runtime_lock:
            # TextWorld 1.7: keyword is request_infos (not infos)
            self._env = self._textworld.start(
                game_file, request_infos=self._request_infos
            )

            # env.reset() returns a GameState
            result = self._env.reset()

        if isinstance(result, tuple):
            self._game_state = result[0]
        else:
            self._game_state = result

        self._mission = (
            getattr(self._game_state, "objective", None)
            or "Find and take the coin."
        )
        return self._to_state(self._game_state)

    def get_instruction_prompt(self, mission: Optional[str] = None) -> str:
        from textworld_env import get_instruction_prompt
        return get_instruction_prompt(mission=mission or self._mission)

    def validate_action(self, action: str) -> tuple[str, bool, str]:
        """Match *action* against current admissible commands.

        Returns ``(action_to_use, was_valid, reason)``.
        """
        a = (action or "").strip()
        a_lower = a.lower()

        # 1. Exact match (case-insensitive)
        for cmd in self._current_valid_actions:
            if cmd.lower() == a_lower:
                return cmd, True, ""

        # 2. Substring / containment match (handles minor LLM wording quirks)
        for cmd in self._current_valid_actions:
            if cmd.lower() in a_lower or a_lower in cmd.lower():
                return cmd, True, ""

        if self.invalid_action_mode == "strict":
            raise ValueError(
                f"Invalid action '{action}'.  Valid: {self._current_valid_actions}"
            )
        return self.fallback_action, False, f"invalid_action:{action}"

    def step(self, action: str) -> StepResult:
        action_used, was_valid, reason = self.validate_action(action)

        prev_score = getattr(self._game_state, "score", 0) or 0

        with _tw_runtime_lock:
            # TextWorld 1.7: step() returns (GameState, score: int, done: bool)
            result = self._env.step(action_used)

        if isinstance(result, tuple):
            game_state = result[0]
            score = result[1] if len(result) > 1 else (getattr(game_state, "score", 0) or 0)
            done = result[2] if len(result) > 2 else False
        else:
            game_state = result
            score = getattr(game_state, "score", 0) or 0
            done = False

        self._game_state = game_state
        delta_reward = score - prev_score

        won = getattr(game_state, "won", False)
        lost = getattr(game_state, "lost", False)
        if not done:
            done = won or lost

        terminated = bool(done and won)
        truncated = bool(done and not won)

        return StepResult(
            state=self._to_state(game_state),
            reward=float(delta_reward),
            terminated=terminated,
            truncated=truncated,
            action_used=action_used,
            action_was_valid=was_valid,
            reason=reason,
        )

    def get_agent_grid_position(self) -> tuple[int, int] | None:
        """TextWorld is graph-based, not grid-based — always ``None``."""
        return None

    def close(self) -> None:
        if self._env is not None:
            try:
                with _tw_runtime_lock:
                    self._env.close()
            except Exception:
                pass
            self._env = None