import re
from typing import List

from memory.schemas import EpisodeMemory, RetrievalHit, StepMemory


class LexicalRetriever:
    """
    Baseline retrieval using token overlap.
    """

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    def score(self, query: str, candidate: str) -> float:
        q = self._tokens(query)
        c = self._tokens(candidate)
        if not q or not c:
            return 0.0
        intersection = len(q.intersection(c))
        union = len(q.union(c))
        return intersection / union

    def retrieve_steps(
        self,
        query_mission: str,
        query_text_obs: str,
        episodes: List[EpisodeMemory],
        top_k: int = 3,
    ) -> List[RetrievalHit]:
        query = f"{query_mission} {query_text_obs}"
        scored: List[RetrievalHit] = []
        for ep in episodes:
            for step in ep.trajectory:
                cand = f"{step.mission} {step.text_obs} {step.action}"
                s = self.score(query, cand)
                if s <= 0:
                    continue
                scored.append(
                    RetrievalHit(
                        score=s,
                        episode_id=step.episode_id,
                        mission=step.mission,
                        text_obs=step.text_obs,
                        action=step.action,
                        reward=step.reward,
                    )
                )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def format_hits(hits: List[RetrievalHit]) -> str:
        if not hits:
            return ""
        lines = ["Relevant past experience:"]
        for i, h in enumerate(hits, start=1):
            lines.append(
                f"{i}. ep={h.episode_id}, mission={h.mission}, obs={h.text_obs}, action={h.action}, reward={h.reward}"
            )
        return "\n".join(lines)
