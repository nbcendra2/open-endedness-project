import re
from typing import List

from memory.schemas import EpisodeMemory, RetrievalHit
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer


class LexicalRetriever:
    """
    Baseline retrieval using token overlap at episode level.
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

    def retrieve_episodes(
        self,
        query_mission: str,
        query_text_obs: str,
        episodes: List[EpisodeMemory],
        top_k: int = 3,
    ) -> List[RetrievalHit]:
        query = f"{query_mission} {query_text_obs}"
        scored: List[RetrievalHit] = []
        for ep in episodes:
            ep_text = f"{ep.mission} {ep.summary} {ep.strategy} {' '.join(ep.lessons)}".strip()
            s = self.score(query, ep_text)
            if s <= 0:
                continue
            scored.append(
                RetrievalHit(
                    score=s,
                    episode_id=ep.episode_id,
                    mission=ep.mission,
                    success=ep.success,
                    total_reward=ep.total_reward,
                    summary=ep.summary,
                    strategy=ep.strategy,
                    lesson=ep.lessons[0] if ep.lessons else "",
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
            outcome = "success" if h.success else "failed"
            parts = [f"ep={h.episode_id} ({outcome}, reward={h.total_reward})"]
            if h.summary:
                parts.append(h.summary)
            if h.strategy:
                parts.append(f"strategy: {h.strategy}")
            if h.lesson:
                parts.append(f"lesson: {h.lesson}")
            lines.append(f"{i}. {', '.join(parts)}")
        return "\n".join(lines)


class EmbeddingRetriever:
    """
    Retrieval using embedding cosine similarity at episode level.
    """
    _MODEL_CACHE = {}
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if model_name not in self._MODEL_CACHE:
            self._MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        self.model = self._MODEL_CACHE[model_name]

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)
    
    def score(self, query_emb: np.ndarray, cand_emb: np.ndarray) -> float:
        # cosine similarity (since normalized)
        return float(np.dot(query_emb, cand_emb))

    def retrieve_episodes(
        self,
        query_mission: str,
        query_text_obs: str,
        episodes: List[EpisodeMemory],
        top_k: int = 3,
    ) -> List[RetrievalHit]:
        query = f"{query_mission} {query_text_obs}"
        query_emb = self.embed(query)

        scored: List[RetrievalHit] = []
        embeddings: List[np.ndarray] = []

        for ep in episodes:
            ep_text = f"{ep.mission} {ep.summary} {ep.strategy} {' '.join(ep.lessons)}".strip()
            ep_emb = self.embed(ep_text)
            s = self.score(query_emb, ep_emb) + (0.1 * ep.total_reward)

            scored.append(
                RetrievalHit(
                    score=s,
                    episode_id=ep.episode_id,
                    mission=ep.mission,
                    success=ep.success,
                    total_reward=ep.total_reward,
                    summary=ep.summary,
                    strategy=ep.strategy,
                    lesson=ep.lessons[0] if ep.lessons else "",
                )
            )
            embeddings.append(ep_emb)

        if len(scored) == 0:
            return []

        N = min(len(scored), top_k * 10)
        idx = np.argsort([-h.score for h in scored])[:N]

        top_hits = [scored[i] for i in idx]
        top_embs = np.array([embeddings[i] for i in idx])

        if len(top_hits) <= top_k:
            return top_hits

        kmeans = KMeans(n_clusters=top_k, n_init=10)
        labels = kmeans.fit_predict(top_embs)

        diverse_hits = []
        for cluster_id in range(top_k):
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            if not cluster_indices:
                continue
            best_idx = max(cluster_indices, key=lambda i: top_hits[i].score)
            diverse_hits.append(top_hits[best_idx])

        diverse_hits.sort(key=lambda x: x.score, reverse=True)
        return diverse_hits[:top_k]

    @staticmethod
    def format_hits(hits: List[RetrievalHit]) -> str:
        if not hits:
            return ""
        lines = ["Relevant past experience:"]
        for i, h in enumerate(hits, start=1):
            outcome = "success" if h.success else "failed"
            parts = [f"ep={h.episode_id} ({outcome}, reward={h.total_reward})"]
            if h.summary:
                parts.append(h.summary)
            if h.strategy:
                parts.append(f"strategy: {h.strategy}")
            if h.lesson:
                parts.append(f"lesson: {h.lesson}")
            lines.append(f"{i}. {', '.join(parts)}")
        return "\n".join(lines)