import json
import os
from dataclasses import asdict
from typing import List

from memory.schemas import EpisodeMemory, StepMemory


class EpisodicMemory:
    """
    Stores long-term memory across episodes.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.episodes: List[EpisodeMemory] = []
        self.load()

    def add_episode(self, episode: EpisodeMemory) -> None:
        self.episodes.append(episode)

    def all_episodes(self) -> List[EpisodeMemory]:
        return self.episodes

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([asdict(x) for x in self.episodes], f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not os.path.exists(self.path):
            self.episodes = []
            return
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.episodes = []
        for item in raw if isinstance(raw, list) else []:
            trajectory = [StepMemory(**s) for s in item.get("trajectory", [])]
            self.episodes.append(
                EpisodeMemory(
                    episode_id=int(item.get("episode_id", 0)),
                    mission=str(item.get("mission", "")),
                    success=bool(item.get("success", False)),
                    total_reward=float(item.get("total_reward", 0.0)),
                    num_steps=int(item.get("num_steps", len(trajectory))),
                    trajectory=trajectory,
                    metadata=dict(item.get("metadata", {})),
                )
            )
