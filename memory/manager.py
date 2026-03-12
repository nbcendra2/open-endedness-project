from typing import List, Optional

from memory.episodic_memory import EpisodicMemory
from memory.retrieval import LexicalRetriever, EmbeddingRetriever
from memory.schemas import EpisodeMemory, RetrievalHit
from memory.working_memory import WorkingMemory


class MemoryManager:
    """
    High-level baseline memory coordinator.
    """

    def __init__(self, episodic_path: str, retrieval_top_k: int = 3, retriever: str = "embedding") -> None:
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(path=episodic_path)
        self.retriever = LexicalRetriever() if retriever == "lexical" else EmbeddingRetriever()
        self.retrieval_top_k = int(retrieval_top_k)

    def start_episode(self, episode_id: int, mission: str) -> None:
        self.working.start_episode(episode_id=episode_id, mission=mission)
        self._preload_insights(mission)

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
        """
        Record the step in the working memory. 
        Update memory
        """
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

    def finish_episode(
        self,
        total_reward: float,
        success: bool,
        summary: str = "",
        strategy: str = "",
        lessons: Optional[List[str]] = None,
    ) -> EpisodeMemory:
        episode_id = self.working.current_episode_id
        if episode_id is None:
            raise RuntimeError("Cannot finish episode before start_episode().")
        ep = EpisodeMemory(
            episode_id=episode_id,
            mission=self.working.current_mission,
            success=bool(success),
            total_reward=float(total_reward),
            num_steps=len(self.working.steps),
            summary=summary,
            strategy=strategy,
            lessons=lessons if lessons is not None else [],
        )

        self.episodic.add_episode(ep)
        self.episodic.save()
        self.working.clear()
        return ep

    def retrieve(self, mission: str, text_obs: str, top_k: int | None = None) -> List[RetrievalHit]:
        """
        Retrieve relevant past steps as memory based on the current mission and text observation.
        Top k retrieval
        """
        k = self.retrieval_top_k if top_k is None else int(top_k)
        return self.retriever.retrieve_episodes(
            query_mission=mission,
            query_text_obs=text_obs,
            episodes=self.episodic.all_episodes(),
            top_k=k,
        )

    def retrieve_as_text(self, mission: str, text_obs: str, top_k: int | None = None) -> str:
        """
        Builds top k RetrievalHit objects and includes working memory insights.
        """
        hits = self.retrieve(mission=mission, text_obs=text_obs, top_k=top_k)
        parts = []

        # Include insights from past episodes
        insights_text = self._format_insights()
        if insights_text:
            parts.append(insights_text)

        # Include objects seen earlier this episode
        seen_text = self.working.format_seen_objects()
        # print(f"[DEBUG retrieve_as_text] seen_objects block: {repr(seen_text)}")
        if seen_text:
            parts.append(seen_text)

        # Include plan if set
        if self.working.plan:
            parts.append(f"Current plan: {self.working.plan}")

        # Include retrieval hits
        hits_text = self.retriever.format_hits(hits)
        if hits_text:
            parts.append(hits_text)

        return "\n\n".join(parts)

    # ── Internal helpers ──────────────────────────────────────────

    def _preload_insights(self, mission: str) -> None:
        """Retrieve lessons from past episodes and load into working memory."""
        hits = self.retrieve(mission=mission, text_obs="")
        for h in hits:
            if h.lesson:
                self.working.add_insight(h.lesson)
            elif h.summary:
                prefix = "Succeeded" if h.success else "Failed"
                self.working.add_insight(f"{prefix}: {h.summary}")

    def _format_insights(self) -> str:
        if not self.working.insights:
            return ""
        lines = ["Lessons from past experience:"]
        for i, insight in enumerate(self.working.insights, start=1):
            lines.append(f"{i}. {insight}")
        return "\n".join(lines)
