import re
from typing import List

from memory.schemas import EpisodeMemory, RetrievalHit, StepMemory
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer


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


class EmbeddingRetriever:
    """
    Retrieval using embedding cosine similarity.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)
    
    def score(self, query_emb: np.ndarray, cand_emb: np.ndarray) -> float:
        # cosine similarity (since normalized)
        return float(np.dot(query_emb, cand_emb))

    def retrieve_steps(self,
        query_mission: str,
        query_text_obs: str,
        episodes: List[EpisodeMemory],
        top_k: int = 3)-> List[RetrievalHit]:
        
        query = f"{query_mission} {query_text_obs}"
        query_emb = self.embed(query)

        scored: List[RetrievalHit] = []
        embeddings = []
        
        for ep in episodes:
            for step in ep.trajectory:
                
                cand = f"{step.mission} {step.text_obs} {step.action}"
                cand_emb = self.embed(cand)
                
                s = self.score(query_emb, cand_emb) + (0.2 * step.reward)

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

                embeddings.append(cand_emb)

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
            lines.append(
                f"{i}. ep={h.episode_id}, mission={h.mission}, obs={h.text_obs}, action={h.action}, reward={h.reward}"
            )
        return "\n".join(lines)