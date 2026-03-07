from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import minigrid

from babyai_text.clean_lang_wrapper import BabyAITextCleanLangWrapper
from babyai_text import get_instruction_prompt as build_instruction_prompt

@dataclass
class EnvState:
    mission: str
    text_obs: str
    valid_actions: list[str]
    raw_obs: Dict[str, Any]
    raw_info: Dict[str, Any]

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
    def __init__(
        self,
        env_name: str = "BabyAI-MixedTrainLocal-v0",
        gym_kwargs: Optional[dict] = None,
        invalid_action_mode: str = "fallback",  # "strict" | "fallback"
        fallback_action: str = "turn left",
    ):
        minigrid.register_minigrid_envs()
        kwargs = gym_kwargs or {"num_dists": 0}
        base_env = gym.make(env_name, **kwargs)
        self.env = BabyAITextCleanLangWrapper(base_env)
        self.invalid_action_mode = invalid_action_mode
        self.fallback_action = fallback_action

    def _to_state(self, obs: Dict[str, Any], info: Dict[str, Any]) -> EnvState:
        mission = str(obs.get("mission", ""))
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
        resolved_mission = mission
        return build_instruction_prompt(self.env, mission=resolved_mission)

    def validate_action(self, action: str) -> tuple[str, bool, str]:
        a = (action or "").strip().lower()
        valid = list(self.env.language_action_space)
        if a in valid:
            return a, True, ""
        if self.invalid_action_mode == "strict":
            raise ValueError(f"Invalid action '{action}'. Valid: {valid}")
        return self.fallback_action, False, f"invalid_action:{action}"

    def step(self, action: str) -> StepResult:
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

    def close(self) -> None:
        self.env.close()
