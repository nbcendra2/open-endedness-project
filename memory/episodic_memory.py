import json
import os
from dataclasses import asdict
from typing import List

from memory.schemas import EpisodeMemory


class EpisodicMemory:
    """
    Stores long-term memory across episodes.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.episodes: List[EpisodeMemory] = []
        self._next_unsaved_idx = 0
        self.load()

    def add_episode(self, episode: EpisodeMemory) -> None:
        self.episodes.append(episode)

    def all_episodes(self) -> List[EpisodeMemory]:
        return self.episodes

    def save(self) -> None:
        if self._next_unsaved_idx >= len(self.episodes):
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            for episode in self.episodes[self._next_unsaved_idx :]:
                json.dump(asdict(episode), f, ensure_ascii=False)
                f.write("\n")
        self._next_unsaved_idx = len(self.episodes)

    def load(self) -> None:
        if not os.path.exists(self.path):
            self.episodes = []
            self._next_unsaved_idx = 0
            return

        self.episodes = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    self.episodes.append(
                        EpisodeMemory(
                            episode_id=int(item.get("episode_id", 0)),
                            mission=str(item.get("mission", "")),
                            success=bool(item.get("success", False)),
                            total_reward=float(item.get("total_reward", 0.0)),
                            num_steps=int(item.get("num_steps", 0)),
                            summary=str(item.get("summary", "")),
                            strategy=str(item.get("strategy", "")),
                            lessons=list(item.get("lessons", [])),
                        )
                    )
        self._next_unsaved_idx = len(self.episodes)
