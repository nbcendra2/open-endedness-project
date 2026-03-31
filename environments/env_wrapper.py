"""
Functionality:

The evaluator and agents should not deal with raw Gymnasium internals (integer actions,
nested obs dicts, Minigrid registration, MixedTrain resampling). EnvWrapper is the single
adapter that sits between Gymnasium/BabyAI and the rest of the project:

- It builds the correct Gym env (including the MixedTrain "/goal" resampling trick).
- It applies BabyAITextCleanLangWrapper so actions are English strings and observations
  carry text.
- It normalises every reset/step output into simple dataclasses (EnvState, StepResult) that
  the agent and evaluator consume.
- It validates LLM-proposed actions and can silently replace invalid ones with a fallback.

The public entry point is environments.make_env (in __init__.py), which just calls
EnvWrapper with the right arguments.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import minigrid

from babyai_text.clean_lang_wrapper import BabyAITextCleanLangWrapper
from babyai_text import get_instruction_prompt as build_instruction_prompt


# Returned by reset() and inside every StepResult so the agent always sees
# the same shape: mission string, text observation, list of valid action strings,
# plus the raw Gym obs/info for debugging or logging.
@dataclass
class EnvState:
    mission: str
    text_obs: str
    valid_actions: list[str]
    raw_obs: Dict[str, Any]
    raw_info: Dict[str, Any]


# Returned by step(). Bundles the new state with reward, termination flags,
# which action was actually executed, whether it was valid, and an optional
# reason string (e.g. "invalid_action:junp forward").
@dataclass
class StepResult:
    state: EnvState
    reward: float
    terminated: bool
    truncated: bool
    action_used: str
    action_was_valid: bool
    reason: str = ""


class EnvWrapper:
    def __init__(self, env_name="BabyAI-MixedTrainLocal-v0", gym_kwargs=None, 
                 invalid_action_mode: str = "fallback", fallback_action: str = "turn left",):
        # Register Minigrid/BabyAI env specs so gym.make can find them
        minigrid.register_minigrid_envs()

        kwargs = dict(gym_kwargs) if gym_kwargs is not None else {}
        # Some BabyAI environments do not accept num_dist(s) as constructor kwargs;
        # remove them so gym.make does not crash on those envs
        kwargs.pop("num_dists", None)
        kwargs.pop("num_dist", None)

        if env_name.startswith("BabyAI-MixedTrainLocal-v0/"):
            # MixedTrain pseudo-id: "BabyAI-MixedTrainLocal-v0/goto" means resample
            # until the mission family matches "goto". Each gym.make draws a random
            # mission type; we keep trying (up to 2000 times) and close rejected
            # candidates to avoid resource leaks.
            base_id, goal = env_name.split("/", 1)
            base_env = None
            for _ in range(2000):
                cand = gym.make(base_id, **kwargs)
                kind = cand.unwrapped.action_kinds[0].replace(" ", "_")
                if kind == goal:
                    base_env = cand
                    break
                cand.close()
            if base_env is None:
                raise RuntimeError(f"Could not sample task '{goal}' from {base_id}")
        else:
            # Plain Gym id like "BabyAI-GoToLocal-v0", one gym.make is enough
            base_env = gym.make(env_name, **kwargs)

        # Wrap so actions are English strings and obs carries text + image
        self.env = BabyAITextCleanLangWrapper(base_env)
        self.invalid_action_mode = invalid_action_mode
        self.fallback_action = fallback_action

    def _to_state(self, obs: Dict[str, Any], info: Dict[str, Any]) -> EnvState:
        """Convert raw Gym obs/info into an EnvState the agent understands."""
        mission = str(obs.get("mission", ""))

        # The wrapper puts a text dict with long_term_context into obs["text"].
        # If that is missing for some reason, fall back to raw descriptions from info.
        text_obs = ""
        text_block = obs.get("text")
        if isinstance(text_block, dict):
            text_obs = str(text_block.get("long_term_context", ""))
        if not text_obs:
            desc = info.get("descriptions") or []
            text_obs = "\n".join(str(x) for x in desc)

        return EnvState(
            mission=mission,
            text_obs=text_obs,
            valid_actions=list(self.env.language_action_space),
            raw_obs=obs,
            raw_info=info,
        )

    def reset(self, seed: Optional[int] = None) -> EnvState:
        obs, info = self.env.reset(seed=seed)
        return self._to_state(obs, info)

    def get_instruction_prompt(self, mission: Optional[str] = None) -> str:
        """Build the system prompt for the LLM (mission + action list + tips)"""
        resolved_mission = mission
        return build_instruction_prompt(self.env, mission=resolved_mission)

    def validate_action(self, action: str) -> tuple[str, bool, str]:
        """
        Check whether the LLM's proposed action string is valid

        Returns (action_to_use, was_valid, reason)
        In "strict" mode an invalid action raises ValueError
        In "fallback" mode it is silently replaced by self.fallback_action
        """
        a = (action or "").strip().lower()
        valid = list(self.env.language_action_space)
        if a in valid:
            return a, True, ""
        if self.invalid_action_mode == "strict":
            raise ValueError(f"Invalid action '{action}'. Valid: {valid}")
        return self.fallback_action, False, f"invalid_action:{action}"

    def step(self, action: str) -> StepResult:
        """Validate the action, execute it in the env, return a StepResult"""
        action_used, ok, reason = self.validate_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_used)
        return StepResult(
            state=self._to_state(obs, info),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            action_used=action_used,
            action_was_valid=ok,
            reason=reason,
        )

    def get_agent_grid_position(self) -> tuple[int, int] | None:
        """Minigrid/BabyAI agent cell ``(column, row)`` after the latest reset/step, or None

        Used for transition metrics (e.g. blocked forward). Non-grid envs should return None.
        """
        try:
            inner = self.env.unwrapped
            pos = getattr(inner, "agent_pos", None)
            if pos is None:
                return None
            return (int(pos[0]), int(pos[1]))
        except Exception:
            return None

    def close(self) -> None:
        self.env.close()
