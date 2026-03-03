"""
Text Memento Agent - Case-based reasoning for BabyAI tasks
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
from typing import List, Dict, Any, Optional
import json
import os


class TextMementoAgent:
    """Agent that retrieves and votes on actions from similar past episodes."""
    
    def __init__(self, archive_path: str, embedding_model: str = "all-MiniLM-L6-v2", top_k: int = 5, seed: int = 0):
        self.archive_path = archive_path
        self.top_k = top_k
        self.encoder = SentenceTransformer(embedding_model)
        self.archive = self._load_archive()
        
    def _load_archive(self) -> List[Dict[str, Any]]:
        """Load archive from JSON file."""
        if not os.path.exists(self.archive_path):
            raise FileNotFoundError(f"Archive not found: {self.archive_path}")
        
        with open(self.archive_path, 'r') as f:
            data = json.load(f)
        
        archive = data.get('cases', [])
        print(f"✅ Loaded archive: {len(archive)} cases")
        return archive
    
    def reset(self, seed: Optional[int] = None):
        """Reset agent state (no-op for stateless retrieval)."""
        pass
    
    def act(self, text_obs: str, mission: str, valid_actions: List[str], step_idx: int) -> Dict[str, Any]:
        """
        Select action by retrieving similar cases and voting.
        
        Args:
            text_obs: Current observation text
            mission: Mission description
            valid_actions: List of valid actions
            step_idx: Current step index in episode
            
        Returns:
            Dictionary with 'action' and 'reason'
        """
        if not self.archive:
            return {"action": "go forward", "reason": "empty archive"}
        
        # Encode current mission
        query_embedding = self.encoder.encode(mission, convert_to_numpy=True)
        
        # Find similar cases
        similar_cases = self._retrieve_similar_cases(query_embedding)
        
        # Vote on action at current step
        action, reason = self._vote_on_action(similar_cases, step_idx, valid_actions)
        
        return {"action": action, "reason": reason}
    
    def _retrieve_similar_cases(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Retrieve top-k similar cases by cosine similarity."""
        similarities = []
        
        for case in self.archive:
            case_emb = np.array(case['embedding'])
            # Cosine similarity
            sim = np.dot(query_embedding, case_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(case_emb)
            )
            similarities.append((sim, case))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k
        top_cases = [case for _, case in similarities[:self.top_k]]
        return top_cases
    
    def _vote_on_action(
        self, 
        cases: List[Dict[str, Any]], 
        step_idx: int,
        valid_actions: List[str]
    ) -> tuple[str, str]:
        """Vote on action based on what similar cases did at this step."""
        votes = []
        
        for case in cases:
            trajectory = case['trajectory']
            # Use action at same step index if available
            if step_idx < len(trajectory):
                action = trajectory[step_idx]['action']
                votes.append(action)
        
        if not votes:
            # No cases have this many steps, use default
            return "go forward", f"no votes at step {step_idx}"
        
        # Count votes
        vote_counts = Counter(votes)
        winning_action = vote_counts.most_common(1)[0][0]
        
        # Validate action
        if winning_action not in valid_actions:
            winning_action = "go forward"
            reason = f"voted action invalid, using fallback"
        else:
            reason = f"voted from {len(votes)} cases at step {step_idx}"
        
        return winning_action, reason