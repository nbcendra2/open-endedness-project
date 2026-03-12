import re
from typing import Dict, List

from memory.schemas import StepMemory

# Matches "a <non-wall object> N step(s) <direction>" lines in text observations
_OBJ_LOC_RE = re.compile(
    r"^(?:a|an)\s+((?!wall\b)[a-z][a-z ]*?)\s+(\d+\s+steps?[^\n]*)$",
    re.IGNORECASE | re.MULTILINE,
)



class WorkingMemory:
    """
    Keeps short-term memory for the current episode only.
    """

    def __init__(self) -> None:
        self.current_episode_id: int | None = None
        self.current_mission: str = ""
        self.steps: List[StepMemory] = []
        self.plan: str = ""
        self.insights: List[str] = []
        self.seen_objects: Dict[str, str] = {}  # obj_name -> "step N (<location>)"

    def start_episode(self, episode_id: int, mission: str) -> None:
        self.current_episode_id = episode_id
        self.current_mission = mission
        self.steps = []
        self.plan = ""
        self.insights = []
        self.seen_objects = {}

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
        self._update_seen_objects(text_obs, step_idx)
        return item

    def _update_seen_objects(self, text_obs: str, step_idx: int) -> None:
        """Parse text_obs for visible objects and record their last known location."""
        matches = list(_OBJ_LOC_RE.finditer(text_obs))
        #print(f"[DEBUG seen_objects] step={step_idx} | raw_obs={repr(text_obs)} | matches={[(m.group(1), m.group(2)) for m in matches]}")
        for match in matches:
            obj = match.group(1).strip().lower()
            location = match.group(2).strip()
            self.seen_objects[obj] = f"step {step_idx} ({location})"
        #print(f"[DEBUG seen_objects] current seen_objects={self.seen_objects}")

    def format_seen_objects(self) -> str:
        if not self.seen_objects:
            return ""
        lines = ["Objects seen this episode:"]
        for obj, info in self.seen_objects.items():
            lines.append(f"- {obj}: last seen at {info}")
        return "\n".join(lines)

    def recent_steps(self, k: int = 3) -> List[StepMemory]:
        if k <= 0:
            return []
        return self.steps[-k:]

    def set_plan(self, plan: str) -> None:
        self.plan = plan

    def add_insight(self, insight: str) -> None:
        self.insights.append(insight)

    def clear(self) -> None:
        self.current_episode_id = None
        self.current_mission = ""
        self.steps = []
        self.plan = ""
        self.insights = []
        self.seen_objects = {}
