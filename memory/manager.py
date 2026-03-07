from typing import List

from memory.episodic_memory import EpisodicMemory
from memory.retrieval import LexicalRetriever
from memory.schemas import EpisodeMemory, RetrievalHit
from memory.working_memory import WorkingMemory


class MemoryManager:
    """
    High-level baseline memory coordinator.
    Use this class from evaluator/agent code.
    """

    def __init__(self, episodic_path: str, retrieval_top_k: int = 3) -> None:
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(path=episodic_path)
        self.retriever = LexicalRetriever()
        self.retrieval_top_k = int(retrieval_top_k)

    def start_episode(self, episode_id: int, mission: str) -> None:
        self.working.start_episode(episode_id=episode_id, mission=mission)

    def record_step(
        self,
        step_idx: int,
        text_obs: str,
        action: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        action_was_valid: bool = True,
        env_reason: str = "",
    ) -> None:
        self.working.add_step(
            step_idx=step_idx,
            text_obs=text_obs,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            action_was_valid=action_was_valid,
            env_reason=env_reason,
        )

    def finish_episode(self, total_reward: float, success: bool) -> EpisodeMemory:
        episode_id = self.working.current_episode_id
        if episode_id is None:
            raise RuntimeError("Cannot finish episode before start_episode().")
        ep = EpisodeMemory(
            episode_id=episode_id,
            mission=self.working.current_mission,
            success=bool(success),
            total_reward=float(total_reward),
            num_steps=len(self.working.steps),
            trajectory=list(self.working.steps),
        )
        self.episodic.add_episode(ep)
        self.episodic.save()
        self.working.clear()
        return ep

    def retrieve(self, mission: str, text_obs: str, top_k: int | None = None) -> List[RetrievalHit]:
        k = self.retrieval_top_k if top_k is None else int(top_k)
        return self.retriever.retrieve_steps(
            query_mission=mission,
            query_text_obs=text_obs,
            episodes=self.episodic.all_episodes(),
            top_k=k,
        )

    def retrieve_as_text(self, mission: str, text_obs: str, top_k: int | None = None) -> str:
        hits = self.retrieve(mission=mission, text_obs=text_obs, top_k=top_k)
        return self.retriever.format_hits(hits)
