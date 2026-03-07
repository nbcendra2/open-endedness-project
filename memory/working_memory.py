from typing import List

from memory.schemas import StepMemory


class WorkingMemory:
    """
    Keeps short-term memory for the current episode only.
    """

    def __init__(self) -> None:
        self.current_episode_id: int | None = None
        self.current_mission: str = ""
        self.steps: List[StepMemory] = []

    def start_episode(self, episode_id: int, mission: str) -> None:
        self.current_episode_id = episode_id
        self.current_mission = mission
        self.steps = []

    def add_step(
        self,
        step_idx: int,
        text_obs: str,
        action: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        action_was_valid: bool = True,
        env_reason: str = "",
    ) -> StepMemory:
        if self.current_episode_id is None:
            raise RuntimeError("WorkingMemory.start_episode() must be called before add_step().")
        item = StepMemory(
            episode_id=self.current_episode_id,
            step_idx=step_idx,
            mission=self.current_mission,
            text_obs=text_obs,
            action=action,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            action_was_valid=bool(action_was_valid),
            env_reason=env_reason,
        )
        self.steps.append(item)
        return item

    def recent_steps(self, k: int = 3) -> List[StepMemory]:
        if k <= 0:
            return []
        return self.steps[-k:]

    def clear(self) -> None:
        self.current_episode_id = None
        self.current_mission = ""
        self.steps = []
