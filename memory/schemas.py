from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StepMemory:
    episode_id: int
    step_idx: int
    mission: str
    text_obs: str
    action: str
    reward: float
    terminated: bool
    truncated: bool
    action_was_valid: bool = True
    env_reason: str = ""


@dataclass
class EpisodeMemory:
    episode_id: int
    mission: str
    success: bool
    total_reward: float
    num_steps: int
    trajectory: List[StepMemory] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    score: float
    episode_id: int
    mission: str
    text_obs: str
    action: str
    reward: float
