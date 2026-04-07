import os
import json
import logging
import re
import math
from dotenv import load_dotenv

from llm_clients import build_llm_client
from prompt_builder import HistoryPromptBuilder

import random as rng
from collections import deque
from typing import List, Dict, Any

load_dotenv()

logger = logging.getLogger(__name__)

# On-disk agent memory: episode reflections (JSON) and optional memory.jsonl-style logs.
# Resolve relative to the project root (parent of agent/) so the path is
# stable regardless of the working directory the script is launched from.
AGENT_MEMORY_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "agent_memory_store",
)

# Reflection persistence helpers

def _task_id_from_env_name(env_name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(env_name))


def _reflections_path(task_id: str) -> str:
    os.makedirs(AGENT_MEMORY_STORE_DIR, exist_ok=True)
    return os.path.join(AGENT_MEMORY_STORE_DIR, f"reflections_{task_id}.json")


def load_reflections(env_name: str, max_reflections: int = 3) -> list:
    task_id = _task_id_from_env_name(env_name)
    path = _reflections_path(task_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return [str(x) for x in data[-max_reflections:]]
    except Exception:
        return []


def append_reflection(env_name: str, reflection_text: str, max_reflections: int = 3) -> None:
    task_id = _task_id_from_env_name(env_name)
    path = _reflections_path(task_id)
    reflections = load_reflections(env_name, max_reflections=max_reflections)
    reflections.append(str(reflection_text))
    reflections = reflections[-max_reflections:]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(reflections, f, ensure_ascii=False, indent=2)
    except OSError:
        logger.warning("Failed to persist reflection to %s", path, exc_info=True)


_VALID_MEMORY_TYPES = frozenset({
    "random", "baseline", "trajectory", "reflection",
    "enriched", "enriched_history", "fade_enriched_history",
})

# Defaults merged with config `agent.params.fade_enriched_history` (fade mode only)
_DEFAULT_FADE_ENRICHED_HISTORY_CFG: Dict[str, Any] = {
    # "text": match trigger_condition to observation via token overlap (no embed(trigger))
    # "embedding": cosine(trigger_embedding, obs_embedding) — requires embed(trigger)
    "trigger_match": "text",
    "trigger_text_threshold": 0.55,
    # None = no cap (all candidates). 0 = skip LLM conflict on embedding candidates only.
    "max_conflict_llm_candidates": None,
    # Package A (API vs. responsiveness): intervals + fusion clustering
    "priming_interval": 6,
    "tag_batch_interval": 12,
    "fusion_interval": 12,
    "theta_fusion": 0.82,
    "t_window": 10,
}


class BaseAgent:
    """Base class for LLM-based agents"""

    MAX_ENRICHED_FACTS = 16
    ENRICHMENT_WINDOW_K = 3
    MAX_REFLECTION_STEPS_FOR_SUMMARY = 16

    # Fade-enriched lifecycle thresholds
    DORMANT_THRESHOLD = 0.15
    DROP_THRESHOLD = 0.05
    REACTIVATION_BOOST = 0.3
    REACTIVATION_THRESHOLD = 0.25
    CONTRADICTION_PENALTY = 0.5
    MAX_DORMANT_AGE = 15

    # Importance score weights (Paper Eq. 2: I = α·rel + β·freq + γ·recency)
    FADE_ALPHA = 0.5       # semantic relevance weight
    FADE_BETA = 0.3        # access frequency weight
    FADE_GAMMA = 0.2       # recency weight
    FADE_LAMBDA_BASE = 0.1   # λ_base: baseline decay rate (Paper §3.1)
    FADE_MU = 3.0            # μ: importance modulation on decay rate
    FADE_KAPPA = 0.1         # κ: temporal decay for access rate f̃_i (Paper Eq. 2)
    FADE_THETA_PROMOTE = 0.7 # θ_promote: threshold to promote SML → LML
    FADE_THETA_DEMOTE = 0.3  # θ_demote: threshold to demote LML → SML

    FADE_DELTA_RECENCY = 0.1   # δ: temporal decay rate for recency (Paper Eq. 2)

    # Memory consolidation on access (Paper Eq. 7)
    FADE_DELTA_V = 0.2       # Δv: base reinforcement strength
    FADE_N = 5               # N: diminishing returns scale
    FADE_W = 10              # W: sliding window size (steps)

    # Conflict resolution (Paper §2.3)
    FADE_THETA_SIM = 0.65    # θ_sim: cosine similarity threshold for candidate retrieval
    FADE_OMEGA = 0.4         # ω: redundancy penalty for compatible memories (Eq. 9)
    FADE_RHO = 1.0           # ρ: suppression strength for contradictory memories (Eq. 10)
    FADE_W_STEP = 30         # W_step: step window for age normalization in contradiction (Eq. 10)

    # Embedding-based reactivation / semantic priming
    FADE_PRIMING_THRESHOLD = 0.5    # cosine similarity threshold for priming to fire
    FADE_PRIMING_BOOST = 0.15       # strength boost for weak active facts, scaled by similarity
    FADE_PRIMING_COOLDOWN = 3       # min steps between priming boosts for same fact
    FADE_PRIMING_INTERVAL = 3       # run priming every N steps
    FADE_PRIMING_WEAK_V = 0.4       # only prime active facts weaker than this

    # Predictive memory tagging (prospective memory)
    FADE_TAG_DECAY_SHIELD = 0.05      # shielded facts decay at this reduced rate
    FADE_TAG_MAX_SHIELD_STEPS = 30    # safety cap: shield expires after this many steps
    FADE_TAG_TRIGGER_THRESHOLD = 0.55 # cosine-sim threshold for trigger activation
    FADE_TAG_TRIGGER_BOOST = 0.2      # v_i boost when trigger fires
    FADE_TAG_BATCH_INTERVAL = 6       # run batch LLM tagging every N steps

    # Adaptive memory fusion (Paper §2.4)
    FADE_THETA_FUSION = 0.75   # θ_fusion: semantic coherence threshold for clustering (Eq. 11)
    FADE_T_WINDOW = 15         # T_window: max step distance for temporal locality (Eq. 11)
    FADE_FUSION_MIN_CLUSTER = 2  # minimum cluster size to trigger fusion
    FADE_FUSION_INTERVAL = 6   # run fusion every N steps
    FADE_FUSION_EPSILON = 0.1  # ε: variance bonus for fused strength (Eq. 12)
    FADE_THETA_PRESERVE = 0.7  # θ_preserve: info preservation threshold for fusion validation (§2.4)

    def __init__(
        self,
        model="gpt-4o-mini",
        seed=0,
        temperature=0.2,
        timeout=10,
        system_prompt=None,
        memory_type: str = "baseline",
        provider: str = "openai",
        fade_enriched_history_cfg: Dict[str, Any] | None = None,
    ):
        self.rng = rng.Random(seed)
        self.memory_type = str(memory_type)
        if self.memory_type not in _VALID_MEMORY_TYPES:
            raise ValueError(
                f"Unknown memory_type {memory_type!r}. "
                f"Choose from: {', '.join(sorted(_VALID_MEMORY_TYPES))}"
            )

        if self.memory_type == "random":
            self.llm = None
            self.prompt_builder = None
            self.temperature = 0.0
            self.timeout = 0.0
            self._enriched_buffer: List[Dict[str, Any]] = []
            self._enrichment_step_buffer: List[Dict[str, Any]] = []
            self._reflection_step_history: List[Dict[str, Any]] = []
            self._env_name: str = ""
            self._fade_enriched_history_cfg: Dict[str, Any] = {}
            return

        self.llm = build_llm_client(provider=provider, model=model)
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        default_system_prompt = (""" 
Your goal is to make progress toward the given mission. First, think about the best course of action given the observations. 
Then you must choose exactly one of the given actions and output it strictly in the following format: 
{
  "reason": "your reasoning in 10 words",
  "action": "YOUR CHOSEN ACTION"
}
Replace YOUR CHOSEN ACTION with the chosen action. Do not output anything outside the JSON.
""")
        self.prompt_builder = HistoryPromptBuilder(
            max_text_history=16,
            system_prompt=system_prompt if system_prompt is not None else default_system_prompt
        )

        self._enriched_buffer: List[Dict[str, Any]] = []
        self._enrichment_step_buffer: List[Dict[str, Any]] = []
        self._reflection_step_history: List[Dict[str, Any]] = []
        self._env_name: str = ""

        raw_fe = dict(fade_enriched_history_cfg or {})
        self._fade_enriched_history_cfg: Dict[str, Any] = (
            {**_DEFAULT_FADE_ENRICHED_HISTORY_CFG, **raw_fe}
            if self.memory_type == "fade_enriched_history"
            else {}
        )
        if self.memory_type == "fade_enriched_history":
            fe = self._fade_enriched_history_cfg
            fe["trigger_match"] = str(fe.get("trigger_match", "text")).lower()
            if fe["trigger_match"] not in ("text", "embedding"):
                fe["trigger_match"] = "text"
            fe["trigger_text_threshold"] = float(fe.get("trigger_text_threshold", 0.55))
            m = fe.get("max_conflict_llm_candidates")
            fe["max_conflict_llm_candidates"] = None if m is None else int(m)
            fe["priming_interval"] = max(1, int(fe.get("priming_interval", 6)))
            fe["tag_batch_interval"] = max(1, int(fe.get("tag_batch_interval", 12)))
            fe["fusion_interval"] = max(1, int(fe.get("fusion_interval", 12)))
            fe["theta_fusion"] = float(fe.get("theta_fusion", 0.82))
            fe["t_window"] = max(1, int(fe.get("t_window", 10)))
            # Instance overrides so act/observe_step use config, not class constants only
            self.FADE_PRIMING_INTERVAL = fe["priming_interval"]
            self.FADE_TAG_BATCH_INTERVAL = fe["tag_batch_interval"]
            self.FADE_FUSION_INTERVAL = fe["fusion_interval"]
            self.FADE_THETA_FUSION = fe["theta_fusion"]
            self.FADE_T_WINDOW = fe["t_window"]

        # Trace logging: opt-in, zero overhead when disabled
        self._trace_enabled: bool = False
        self._trace_log: List[Dict[str, Any]] = []

        # Fade-enriched-only state (unused by other memory modes)
        if self.memory_type == "fade_enriched_history":
            self._current_mission: str = ""
            self._mission_embedding: List[float] | None = None
            self._current_obs_embedding: List[float] | None = None
            self._last_obs_text: str = ""
            self._fade_reactivation_count: int = 0
            self._fade_priming_count: int = 0
            self._fade_tag_trigger_count: int = 0
            self._fade_tag_expire_count: int = 0
            self._fade_contradiction_count: int = 0
            self._fade_compatible_count: int = 0
            self._fade_subsume_count: int = 0
            self._fade_fusion_count: int = 0

    # --- Trace logging helpers ---

    def _trace(self, event_type: str, step_idx: int, **data: Any) -> None:
        if self._trace_enabled:
            self._trace_log.append({"event": event_type, "step": step_idx, **data})

    def get_trace_snapshot(self) -> Dict[str, Any]:
        """Return collected trace events and a final buffer snapshot (no embeddings)."""
        return {
            "trace_events": list(self._trace_log),
            "final_buffer": [
                {k: v for k, v in f.items() if k != "embedding" and k != "trigger_embedding"}
                for f in self._enriched_buffer
            ],
        }

    def get_step_memory_snapshot(self) -> Dict[str, Any]:
        """Lightweight per-step snapshot of the enriched buffer (no embeddings)."""
        active = []
        dormant_count = 0
        for f in self._enriched_buffer:
            state = f.get("state", "active")
            if state == "active":
                active.append({
                    "fact": f.get("fact", ""),
                    "v_i": f.get("v_i"),
                    "steps": f.get("steps", ""),
                    "shield_active": f.get("shield_active", False),
                    "trigger_condition": f.get("trigger_condition", ""),
                })
            else:
                dormant_count += 1
        return {
            "buffer_size": len(self._enriched_buffer),
            "active_facts": active,
            "dormant_count": dormant_count,
        }

    # --- Fade-enriched helper functions ---

    _MEMORY_STOP_WORDS = frozenset({
        "the", "a", "an", "is", "in", "to", "of", "and", "was", "it", "that",
        "for", "on", "are", "with", "as", "at", "by", "from", "or", "be",
        "this", "have", "has", "had", "not", "but", "all", "can", "her",
        "his", "they", "them", "its", "you", "your", "my", "we", "our",
        "i", "me", "do", "did", "no", "so", "if", "up", "am",
    })

    @staticmethod
    def _tokenize_for_memory(text: str) -> set:
        """Extract lowercase word tokens, filtering out stop words and very short tokens"""
        words = re.findall(r'[a-z]+', text.lower())
        return {w for w in words if w not in BaseAgent._MEMORY_STOP_WORDS and len(w) > 1}

    @staticmethod
    def _mission_conditioned_relevance(fact_text: str, mission: str, text_obs: str = "") -> float:
        """Score how relevant a fact is to the current mission and observation

        Returns a value in [0, 1] based on word overlap between the fact
        and the mission tokens, with a smaller contribution from the current observation
        """
        fact_tokens = BaseAgent._tokenize_for_memory(fact_text)
        mission_tokens = BaseAgent._tokenize_for_memory(mission)
        if not fact_tokens or not mission_tokens:
            return 0.0
        mission_overlap = len(fact_tokens & mission_tokens) / len(mission_tokens)
        obs_score = 0.0
        if text_obs:
            obs_tokens = BaseAgent._tokenize_for_memory(str(text_obs))
            if obs_tokens:
                obs_score = len(fact_tokens & obs_tokens) / max(len(obs_tokens), 1)
        return min(1.0, 0.7 * mission_overlap + 0.3 * obs_score)

    @staticmethod
    def _reactivation_score(fact: dict, mission: str, text_obs: str) -> float:
        """Check if a dormant fact should be reactivated based on current context"""
        return BaseAgent._mission_conditioned_relevance(
            fact.get("fact", ""), mission, text_obs,
        )

    @staticmethod
    def _trigger_obs_token_overlap(trigger_condition: str, observation: str) -> float:
        """Share of meaningful trigger tokens that appear in the observation, in [0, 1]."""
        tt = BaseAgent._tokenize_for_memory(str(trigger_condition))
        ot = BaseAgent._tokenize_for_memory(str(observation))
        if not tt:
            return 0.0
        return min(1.0, len(tt & ot) / len(tt))

    @staticmethod
    def _facts_contradict(old_fact_text: str, new_fact_text: str) -> bool:
        """Simple rule-based contradiction detection for BabyAI navigation facts

        Checks for known state-change pairs (open/closed, locked/unlocked, etc)
        and requires at least one shared entity word to avoid false positives
        """
        old_lower = old_fact_text.lower()
        new_lower = new_fact_text.lower()
        contradiction_pairs = [
            ("closed", "open"),
            ("locked", "unlocked"),
            ("picked up", "on the floor"),
            ("picked up", "is located"),
            ("not found", "found"),
            ("not visible", "visible"),
        ]
        for state_a, state_b in contradiction_pairs:
            if (state_a in old_lower and state_b in new_lower) or \
               (state_b in old_lower and state_a in new_lower):
                old_tokens = BaseAgent._tokenize_for_memory(old_fact_text)
                new_tokens = BaseAgent._tokenize_for_memory(new_fact_text)
                entity_words = {w for w in (old_tokens & new_tokens) if len(w) > 2}
                if entity_words:
                    return True
        return False

    def _apply_contradiction_update(self, existing_fact: dict, new_entry: dict) -> None:
        """Weaken an existing fact that is contradicted by a newer one"""
        existing_fact["v_i"] = existing_fact.get("v_i", 1.0) * self.CONTRADICTION_PENALTY
        existing_fact["state"] = "dormant"
        existing_fact["superseded_by"] = new_entry.get("steps", "")
        self._fade_contradiction_count += 1

    # --- Embedding-based conflict resolution (Paper §2.3) ---

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _find_conflict_candidates(self, new_embedding: List[float]) -> List[dict]:
        """Paper Eq. 8: retrieve existing facts with sim > θ_sim"""
        candidates = []
        for f in self._enriched_buffer:
            if f.get("state") != "active" or "embedding" not in f:
                continue
            sim = self._cosine_similarity(new_embedding, f["embedding"])
            if sim > self.FADE_THETA_SIM:
                f["_sim"] = sim
                candidates.append(f)
        return candidates

    def _classify_relationship(self, new_fact_text: str, old_fact_text: str) -> str:
        """LLM-based classification into compatible/contradictory/subsumes/subsumed"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You classify the relationship between two memory facts. "
                    "Output exactly one word: compatible, contradictory, subsumes, or subsumed.\n"
                    "- compatible: both can be true simultaneously\n"
                    "- contradictory: they cannot both be true (state change, correction)\n"
                    "- subsumes: the NEW fact is more general and makes the OLD one redundant\n"
                    "- subsumed: the OLD fact is more general and the NEW one adds no new info"
                ),
            },
            {
                "role": "user",
                "content": f"OLD FACT: {old_fact_text}\nNEW FACT: {new_fact_text}\n\nRelationship:",
            },
        ]
        try:
            raw = self.llm.generate(messages, temperature=0.0, timeout=self.timeout)
            label = raw.strip().lower().rstrip(".")
            if label in ("compatible", "contradictory", "subsumes", "subsumed"):
                self._trace("conflict_classified", -1,
                            old_fact=old_fact_text, new_fact=new_fact_text,
                            label=label, llm_raw=raw.strip())
                return label
        except Exception:
            pass
        self._trace("conflict_classified", -1,
                    old_fact=old_fact_text, new_fact=new_fact_text,
                    label="compatible", llm_raw="(default/error)")
        return "compatible"

    def _merge_facts_via_llm(self, fact_a: str, fact_b: str) -> str:
        """LLM merges two related facts into one consolidated statement"""
        messages = [
            {
                "role": "system",
                "content": (
                    "Merge the two facts into a single concise, self-contained statement. "
                    "Preserve all unique information. Remove redundancy. Output one sentence only."
                ),
            },
            {
                "role": "user",
                "content": f"FACT A: {fact_a}\nFACT B: {fact_b}",
            },
        ]
        try:
            merged = self.llm.generate(messages, temperature=0.0, timeout=self.timeout)
            merged = merged.strip()
            if merged:
                self._trace("facts_merged", -1,
                            fact_a=fact_a, fact_b=fact_b, merged=merged)
                return merged
        except Exception:
            pass
        return fact_a

    def _resolve_conflicts(self, new_entry: dict, step_idx: int) -> None:
        """Full conflict resolution pipeline (Paper §2.3)

        Uses rule-based contradiction as fast path, then embedding + LLM
        for remaining candidates.
        """
        new_text = new_entry.get("fact", "")
        new_emb = new_entry.get("embedding")

        # Fast path: rule-based contradiction (existing upgrade, unchanged)
        for existing in self._enriched_buffer:
            if existing.get("state") == "active" and \
               self._facts_contradict(existing.get("fact", ""), new_text):
                v_before = existing.get("v_i", 1.0)
                self._apply_contradiction_update(existing, new_entry)
                self._trace("conflict_resolved", step_idx,
                            old_fact=existing.get("fact", ""), new_fact=new_text,
                            label="contradictory_rule", v_i_before=v_before,
                            v_i_after=existing.get("v_i"), state=existing.get("state"))

        # Embedding-based: find candidates and classify via LLM
        if new_emb is None:
            return

        candidates = self._find_conflict_candidates(new_emb)
        max_k = self._fade_enriched_history_cfg.get("max_conflict_llm_candidates")
        if max_k is not None:
            if max_k <= 0:
                candidates = []
            else:
                candidates.sort(key=lambda c: c.get("_sim", 0.0), reverse=True)
                candidates = candidates[: int(max_k)]

        for cand in candidates:
            # Skip if already handled by rule-based path
            if cand.get("state") != "active":
                continue

            sim = cand.get("_sim", 0.0)
            label = self._classify_relationship(new_text, cand.get("fact", ""))

            v_before = cand.get("v_i", 1.0)

            if label == "compatible":
                # Paper Eq. 9: Ii = Ii · (1 − ω · sim) — multiplicative importance reduction
                factor = 1.0 - self.FADE_OMEGA * sim
                cand["importance_factor"] = cand.get("importance_factor", 1.0) * factor
                self._fade_compatible_count += 1

            elif label == "contradictory":
                # Paper Eq. 10: temporal suppression with age-normalized penalty
                age_diff = step_idx - int(cand.get("creation_step", 0))
                norm_age = min(1.0, max(0.0, age_diff / self.FADE_W_STEP))
                cand["v_i"] = cand.get("v_i", 1.0) * math.exp(-self.FADE_RHO * norm_age)
                if cand["v_i"] < self.DORMANT_THRESHOLD:
                    cand["state"] = "dormant"
                cand["superseded_by"] = new_entry.get("steps", "")
                self._fade_contradiction_count += 1

            elif label in ("subsumes", "subsumed"):
                # LLM-guided merge: consolidate into one fact
                merged_text = self._merge_facts_via_llm(new_text, cand.get("fact", ""))
                if label == "subsumes":
                    # New is more general → old becomes dormant, new gets merged content
                    cand["state"] = "dormant"
                    cand["superseded_by"] = new_entry.get("steps", "")
                    new_entry["fact"] = merged_text
                    try:
                        new_entry["embedding"] = self.llm.embed(merged_text)
                    except Exception:
                        pass
                else:
                    # Old is more general → new is redundant, enrich old with merged content
                    cand["fact"] = merged_text
                    try:
                        cand["embedding"] = self.llm.embed(merged_text)
                    except Exception:
                        pass
                    cand["v_i"] = min(1.0, cand.get("v_i", 1.0) + 0.1)
                    new_entry["state"] = "dormant"
                self._fade_subsume_count += 1

            self._trace("conflict_resolved", step_idx,
                        old_fact=cand.get("fact", ""), new_fact=new_text,
                        label=label, similarity=sim, v_i_before=v_before,
                        v_i_after=cand.get("v_i"), state=cand.get("state"))

            # Clean up temp field
            cand.pop("_sim", None)

    # --- Adaptive memory fusion (Paper §2.4) ---

    def _fuse_facts_via_llm(self, fact_texts: List[str]) -> str:
        """LLM fuses multiple related facts into one consolidated statement"""
        numbered = "\n".join(f"FACT {i+1}: {t}" for i, t in enumerate(fact_texts))
        messages = [
            {
                "role": "system",
                "content": (
                    "Fuse the following related facts into a single concise, self-contained statement. "
                    "Preserve all unique information and temporal progression. "
                    "Remove redundancy. Output one or two sentences only."
                ),
            },
            {"role": "user", "content": numbered},
        ]
        try:
            fused = self.llm.generate(messages, temperature=0.0, timeout=self.timeout)
            fused = fused.strip()
            if fused:
                return fused
        except Exception:
            pass
        return fact_texts[0]

    def _validate_fusion_preservation(self, original_texts: List[str], fused_text: str) -> bool:
        """Paper §2.4: LLM verification that fused text preserves key information"""
        originals = "\n".join(f"- {t}" for t in original_texts)
        messages = [
            {
                "role": "system",
                "content": (
                    "You verify whether a fused memory preserves the key information "
                    "from the original memories. Rate preservation from 0.0 to 1.0. "
                    "Output only the numeric score, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"ORIGINAL MEMORIES:\n{originals}\n\n"
                    f"FUSED MEMORY: {fused_text}\n\n"
                    "Preservation score (0.0–1.0):"
                ),
            },
        ]
        try:
            raw = self.llm.generate(messages, temperature=0.0, timeout=self.timeout)
            score = float(raw.strip())
            return score >= self.FADE_THETA_PRESERVE
        except Exception:
            return True

    def _build_fusion_clusters(self) -> List[List[dict]]:
        """Paper Eq. 11: temporal-semantic clustering of active facts"""
        active = [
            f for f in self._enriched_buffer
            if f.get("state") == "active" and f.get("embedding") is not None
        ]
        if len(active) < self.FADE_FUSION_MIN_CLUSTER:
            return []

        assigned = set()
        clusters: List[List[dict]] = []

        for i, anchor in enumerate(active):
            if id(anchor) in assigned:
                continue
            cluster = [anchor]
            assigned.add(id(anchor))

            for j, candidate in enumerate(active):
                if j <= i or id(candidate) in assigned:
                    continue
                sim = self._cosine_similarity(anchor["embedding"], candidate["embedding"])
                step_dist = abs(
                    int(anchor.get("creation_step", 0)) - int(candidate.get("creation_step", 0))
                )
                if sim > self.FADE_THETA_FUSION and step_dist < self.FADE_T_WINDOW:
                    cluster.append(candidate)
                    assigned.add(id(candidate))

            if len(cluster) >= self.FADE_FUSION_MIN_CLUSTER:
                clusters.append(cluster)

        return clusters

    def _attempt_memory_fusion(self, step_idx: int) -> None:
        """Periodically fuse semantically similar, temporally close facts (Paper §2.4)"""
        clusters = self._build_fusion_clusters()
        if not clusters:
            return

        for cluster in clusters:
            fact_texts = [f.get("fact", "") for f in cluster]
            fused_text = self._fuse_facts_via_llm(fact_texts)

            # Paper §2.4: reject fusion if information preservation is below threshold
            if not self._validate_fusion_preservation(fact_texts, fused_text):
                self._trace("fusion_rejected", step_idx,
                            original_facts=fact_texts, fused_text=fused_text)
                continue

            # Paper Eq. 12: v_fused = max(vi) + ε · var(vi)
            strengths = [f.get("v_i", 0.5) for f in cluster]
            v_max = max(strengths)
            if len(strengths) > 1:
                mean_v = sum(strengths) / len(strengths)
                variance = sum((s - mean_v) ** 2 for s in strengths) / len(strengths)
            else:
                variance = 0.0
            v_fused = min(1.0, v_max + self.FADE_FUSION_EPSILON * variance)

            # Aggregate access history
            all_access = []
            for f in cluster:
                all_access.extend(f.get("access_steps", []))
            total_f_i = sum(f.get("f_i", 0) for f in cluster)

            earliest_step = min(int(f.get("creation_step", 0)) for f in cluster)
            step_spans = [f.get("steps", "") for f in cluster]
            fused_span = f"{step_spans[0]}~{step_spans[-1]}"

            try:
                fused_emb = self.llm.embed(fused_text)
            except Exception:
                fused_emb = cluster[0].get("embedding")

            fused_entry = {
                "steps": fused_span,
                "fact": fused_text,
                "v_i": v_fused,
                "age_i": 0,
                "f_i": total_f_i,
                "access_steps": all_access,
                "embedding": fused_emb,
                "creation_step": earliest_step,
                # Paper Eq. 13: reduced decay via ξ_fused = 1/(1+log|C|)
                "fusion_decay_factor": 1.0 / (1.0 + math.log(len(cluster))),
                "layer_i": "LML",
                "state": "active",
                "reactivation_hits": 0,
                "superseded_by": "",
                "fused_from": len(cluster),
            }

            # Retire cluster members
            for f in cluster:
                f["state"] = "dormant"
                f["superseded_by"] = f"fused@step{step_idx}"

            self._enriched_buffer.append(fused_entry)
            self._fade_fusion_count += 1
            self._trace("fusion_performed", step_idx,
                        original_facts=fact_texts, fused_text=fused_text,
                        v_fused=v_fused, cluster_size=len(cluster))

    # --- Predictive memory tagging (prospective memory) ---

    _TRIGGER_RULES = [
        # BabyAI: key picked up → will need it at locked door
        {"pattern": r"(picked up|carrying|got|holding|acquired)\b.*?\b(\w+)\s+(key)\b",
         "trigger_template": "agent encounters a locked {attr} door",
         "groups": {"attr": 2}},
        # BabyAI: locked/closed door seen → need matching key
        {"pattern": r"\b(locked|closed)\s+(\w+)\s+(door)\b",
         "trigger_template": "agent picks up the {attr} key",
         "groups": {"attr": 2}},
        # BabyAI: target object located but not yet adjacent
        {"pattern": r"(located|found|visible|spotted|seen)\b.*?\b(\w+\s+\w+)\b.*(ahead|left|right|behind)",
         "trigger_template": "agent is adjacent to the {object}",
         "groups": {"object": 2},
         "mission_filter": r"(go to|pick up)"},
        # TreasureHunter: object inside container/on supporter
        {"pattern": r"\b(?:in|on|inside|from)\s+the\s+(\w+)",
         "trigger_template": "agent is in the room with the {container}",
         "groups": {"container": 1},
         "mission_filter": r"(take|recover|get|find)"},
        # TreasureHunter: locked container or door
        {"pattern": r"\b(locked)\s+(\w+)\b",
         "trigger_template": "agent has the key to unlock the {object}",
         "groups": {"object": 2}},
        # TreasureHunter: key in inventory
        {"pattern": r"\b(key)\b.*\b(inventory|carrying|have)\b",
         "trigger_template": "agent encounters a locked door or container",
         "groups": {}},
        # TreasureHunter: exit/passage with direction
        {"pattern": r"\b(exit|passage)\b.*?\b(north|south|east|west)\b",
         "trigger_template": "agent needs to navigate {direction}",
         "groups": {"direction": 2}},
        # Both: object dropped
        {"pattern": r"\b(dropped|put down|left behind)\b.*?\b(\w+)\b",
         "trigger_template": "agent needs the {object} to complete the mission",
         "groups": {"object": 2}},
    ]

    def _generate_trigger_rule_based(self, fact_text: str, mission: str) -> str | None:
        """Try to generate a trigger condition from regex rules.

        Returns a trigger string if a rule matches, None otherwise.
        """
        fact_lower = fact_text.lower()
        mission_lower = mission.lower()

        for rule in self._TRIGGER_RULES:
            mission_filter = rule.get("mission_filter")
            if mission_filter and not re.search(mission_filter, mission_lower):
                continue

            m = re.search(rule["pattern"], fact_lower)
            if not m:
                continue

            template = rule["trigger_template"]
            groups = rule.get("groups", {})
            try:
                replacements = {}
                for name, group_idx in groups.items():
                    value = m.group(group_idx).strip()
                    if not value:
                        raise IndexError(f"Empty capture for group {group_idx}")
                    replacements[name] = value
                result = template.format(**replacements)
                if not result or result == template:
                    continue
                return result
            except (IndexError, KeyError):
                logger.debug(
                    "Trigger rule %r matched but group extraction failed for fact: %s",
                    rule["pattern"], fact_text,
                )
                continue

        return None

    def _generate_triggers_batch_llm(self, step_idx: int) -> None:
        """Periodically tag unshielded active facts via a single LLM call.

        Collects all active facts without a shield, sends them as a batch
        to the LLM for cross-fact causal reasoning, and sets shields on
        facts the LLM identifies as having future relevance.
        """
        unshielded = [
            f for f in self._enriched_buffer
            if f.get("state") == "active"
            and not f.get("shield_active", False)
            and f.get("embedding") is not None
        ]
        if not unshielded:
            return

        mission = self._current_mission
        env_type = "TextWorld-TreasureHunter" if "treasure" in self._env_name.lower() \
            else "BabyAI-gridworld"

        numbered = "\n".join(
            f"{i+1}. \"{f.get('fact', '')}\" (Step {f.get('creation_step', '?')})"
            for i, f in enumerate(unshielded)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a prospective memory analyst for a navigation agent. "
                    "You decide which memory facts will become critical in a FUTURE situation "
                    "and should be protected from forgetting.\n\n"
                    "For each fact, output one line in this exact format:\n"
                    "  <number>: YES <trigger> OR <number>: NO\n\n"
                    "Rules:\n"
                    "- YES means the fact will be critical later; provide a concrete, "
                    "observable trigger (what the agent will SEE)\n"
                    "- NO means the fact is routine or immediately useful (no protection needed)\n"
                    "- Be specific: 'agent sees a locked red door' not 'agent needs key'\n"
                    "- Consider relationships BETWEEN facts when deciding"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"MISSION: {mission}\n"
                    f"ENVIRONMENT: {env_type}\n\n"
                    f"FACTS:\n{numbered}\n\n"
                    "Output one line per fact:"
                ),
            },
        ]

        try:
            raw = self.llm.generate(messages, temperature=0.0, timeout=self.timeout)
        except Exception:
            return

        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Parse lines like "1: YES agent sees a locked door" or "2: NO"
            m = re.match(r"(\d+)\s*:\s*(YES|NO)\s*(.*)", line, re.IGNORECASE)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            decision = m.group(2).upper()
            trigger_text = m.group(3).strip()

            if decision != "YES" or not trigger_text or idx < 0 or idx >= len(unshielded):
                continue

            fact = unshielded[idx]
            fact["shield_active"] = True
            fact["trigger_condition"] = trigger_text
            fact["shield_start_step"] = step_idx
            if self._fade_enriched_history_cfg.get("trigger_match", "text") == "embedding":
                try:
                    fact["trigger_embedding"] = self.llm.embed(trigger_text)
                except Exception:
                    fact["shield_active"] = False
                    fact.pop("trigger_condition", None)
            else:
                fact["trigger_embedding"] = None
            self._trace("shield_created_llm", step_idx,
                        fact=fact.get("fact", ""), trigger_condition=trigger_text,
                        llm_raw=raw.strip())

    def _check_triggers(
        self,
        obs_embedding: list | None,
        step_idx: int,
        obs_text: str = "",
    ) -> None:
        """Check if any shielded fact's trigger matches the current observation.

        With trigger_match=text: token overlap between trigger_condition and obs_text.
        With trigger_match=embedding: cosine(trigger_embedding, obs_embedding).
        """
        use_text = self._fade_enriched_history_cfg.get("trigger_match", "text") == "text"
        obs_str = str(obs_text)

        for f in self._enriched_buffer:
            if not f.get("shield_active", False):
                continue

            shield_age = step_idx - f.get("shield_start_step", 0)
            if shield_age > self.FADE_TAG_MAX_SHIELD_STEPS:
                f["shield_active"] = False
                self._fade_tag_expire_count += 1
                self._trace("trigger_expired", step_idx,
                            fact=f.get("fact", ""), trigger_condition=f.get("trigger_condition", ""),
                            shield_age=shield_age)
                continue

            if use_text:
                tr = (f.get("trigger_condition") or "").strip()
                if not tr:
                    continue
                thr = float(
                    self._fade_enriched_history_cfg.get(
                        "trigger_text_threshold", self.FADE_TAG_TRIGGER_THRESHOLD
                    )
                )
                score = self._trigger_obs_token_overlap(tr, obs_str)
                if score >= thr:
                    v_before = f.get("v_i", 0.0)
                    f["shield_active"] = False
                    f["v_i"] = min(1.0, v_before + self.FADE_TAG_TRIGGER_BOOST)
                    self._fade_tag_trigger_count += 1
                    self._trace("trigger_fired", step_idx,
                                fact=f.get("fact", ""), trigger_condition=tr,
                                match_mode="text", match_score=score,
                                v_i_before=v_before, v_i_after=f["v_i"])
            else:
                if obs_embedding is None:
                    continue
                trigger_emb = f.get("trigger_embedding")
                if trigger_emb is None:
                    continue
                sim = self._cosine_similarity(trigger_emb, obs_embedding)
                if sim >= self.FADE_TAG_TRIGGER_THRESHOLD:
                    v_before = f.get("v_i", 0.0)
                    f["shield_active"] = False
                    f["v_i"] = min(1.0, v_before + self.FADE_TAG_TRIGGER_BOOST)
                    self._fade_tag_trigger_count += 1
                    self._trace("trigger_fired", step_idx,
                                fact=f.get("fact", ""), trigger_condition=f.get("trigger_condition", ""),
                                match_mode="embedding", match_score=sim,
                                v_i_before=v_before, v_i_after=f["v_i"])

    def extract_action(self, response, valid_actions):
        if response is None:
            return None, "No response"
        
        if isinstance(response, dict):
            action = response.get("action")
            reason = response.get("reason")
            if action not in valid_actions:
                return None, f"Invalid action proposed: {action}"
            return action, reason
        
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
                action = parsed.get("action")
                reason = parsed.get("reason")
                if action not in valid_actions:
                    return None, f"Invalid action proposed: {action}"
                return action, reason
            except json.JSONDecodeError:
                return None, "Invalid JSON format"
            
        return None, f"Unsupported response type: {type(response).__name__}"

    def act(self, text_obs, mission, valid_actions, step_idx):
        if self.memory_type == "random":
            a = self.rng.choice(valid_actions)
            return {"action": a, "reason": "random baseline"}

        if self.memory_type in ("baseline", "reflection"):
            # Baseline: no history, only system prompt + current observation.
            # Reflection: same within-episode behaviour, but start_episode has
            # prepended past reflections to the system prompt and end_episode
            # will generate a new reflection for the next episode.
            messages = [
                {
                    "role": "system",
                    "content": self.prompt_builder.system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        f"Mission: {mission}\n\n"
                        f"Observation: {text_obs}\n\n"
                        f"Valid actions: {valid_actions}\n"
                    ),
                },
            ]
            response = self.llm.generate_action_structured(
                messages,
                temperature=self.temperature,
                timeout=self.timeout,
                valid_actions=valid_actions,
            )
        elif self.memory_type == "enriched":
            # Enriched memory: use compressed facts from previous steps
            facts = self._enriched_buffer[-self.MAX_ENRICHED_FACTS :]
            enriched_lines = []
            if facts:
                enriched_lines.append("Enriched memory from previous steps:")
                for fact in facts:
                    span = fact.get("steps", "")
                    text = fact.get("fact", "")
                    enriched_lines.append(f"Steps {span}: \"{text}\"")
            enriched_block = "\n".join(enriched_lines) if enriched_lines else ""

            system_content = self.prompt_builder.system_prompt
            if enriched_block:
                system_content = f"{system_content}\n\n{enriched_block}"

            messages = [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": (
                        f"Mission: {mission}\n\n"
                        f"Observation: {text_obs}\n\n"
                        f"Valid actions: {valid_actions}\n"
                    ),
                },
            ]
            response = self.llm.generate_action_structured(
                messages,
                temperature=self.temperature,
                timeout=self.timeout,
                valid_actions=valid_actions,
            )
        elif self.memory_type == "enriched_history":
            # Hybrid: trajectory prompt history + enriched memory block
            facts = self._enriched_buffer[-self.MAX_ENRICHED_FACTS :]
            if facts:
                self._trace("facts_selected_for_prompt", step_idx,
                            selected=[{"fact": f.get("fact",""), "steps": f.get("steps","")} for f in facts],
                            total_buffer=len(self._enriched_buffer))
            enriched_lines = []
            if facts:
                enriched_lines.append("Enriched memory from previous steps:")
                for fact in facts:
                    span = fact.get("steps", "")
                    text = fact.get("fact", "")
                    enriched_lines.append(f"Steps {span}: \"{text}\"")
            enriched_block = "\n".join(enriched_lines) if enriched_lines else ""

            self.prompt_builder.update_observation(
                text_obs=text_obs,
                mission=mission,
                step_idx=step_idx,
            )
            messages = self.prompt_builder.build_messages(valid_actions)

            # Attach enriched memory to the latest user turn.
            if enriched_block:
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        messages[i] = dict(messages[i])
                        messages[i]["content"] = (
                            f"{messages[i]['content']}\n\n--- Enriched memory ---\n{enriched_block}"
                        )
                        break

            response = self.llm.generate_action_structured(
                messages,
                temperature=self.temperature,
                timeout=self.timeout,
                valid_actions=valid_actions,
            )
        elif self.memory_type == "fade_enriched_history":
            # Hybrid: trajectory prompt history + fade-enriched memory block

            # Cache observation embedding every FADE_PRIMING_INTERVAL steps
            obs_text_str = str(text_obs)
            if step_idx % self.FADE_PRIMING_INTERVAL == 0 and obs_text_str != self._last_obs_text:
                try:
                    self._current_obs_embedding = self.llm.embed(obs_text_str)
                    self._last_obs_text = obs_text_str
                except Exception:
                    pass

            obs_emb = self._current_obs_embedding

            # Phase 1: reactivation + priming via embedding cosine (token-overlap fallback)
            for f in self._enriched_buffer:
                fact_emb = f.get("embedding")

                if fact_emb is not None and obs_emb is not None:
                    sim = self._cosine_similarity(fact_emb, obs_emb)
                else:
                    sim = self._reactivation_score(f, mission, obs_text_str)

                if f.get("state") == "dormant":
                    if sim >= self.REACTIVATION_THRESHOLD:
                        v_before = f.get("v_i", 0.0)
                        f["state"] = "active"
                        f["v_i"] = min(1.0, v_before + self.REACTIVATION_BOOST)
                        f["age_i"] = 0
                        f["reactivation_hits"] = f.get("reactivation_hits", 0) + 1
                        self._fade_reactivation_count += 1
                        self._trace("reactivation", step_idx,
                                    fact=f.get("fact", ""), similarity=sim,
                                    v_i_before=v_before, v_i_after=f["v_i"])

                elif f.get("state") == "active" and f.get("v_i", 1.0) < self.FADE_PRIMING_WEAK_V:
                    if sim >= self.FADE_PRIMING_THRESHOLD:
                        cooldown_ok = (step_idx - f.get("last_primed_step", -999)) >= self.FADE_PRIMING_COOLDOWN
                        if cooldown_ok:
                            v_before = f["v_i"]
                            f["v_i"] = min(1.0, v_before + self.FADE_PRIMING_BOOST * sim)
                            f["last_primed_step"] = step_idx
                            self._fade_priming_count += 1
                            self._trace("priming_boost", step_idx,
                                        fact=f.get("fact", ""), similarity=sim,
                                        v_i_before=v_before, v_i_after=f["v_i"])

            # Phase 1b: check if any shielded fact's trigger matches current observation
            self._check_triggers(obs_emb, step_idx, obs_text_str)

            # Phase 2: select only active facts, rank by mission relevance + strength
            active_facts = [
                f for f in self._enriched_buffer
                if f.get("state", "active") == "active" and "v_i" in f
            ]
            if active_facts:
                for f in active_facts:
                    f["mission_score"] = self._mission_conditioned_relevance(
                        f.get("fact", ""), mission, str(text_obs),
                    )
                active_facts.sort(
                    key=lambda x: 0.5 * x.get("mission_score", 0.0)
                                + 0.4 * x.get("v_i", 0.0)
                                + 0.1 * min(1.0, x.get("f_i", 0) / 3.0),
                    reverse=True,
                )
                top_facts = active_facts[: self.MAX_ENRICHED_FACTS]
                for f in top_facts:
                    f["f_i"] = f.get("f_i", 0) + 1
                    f["age_i"] = 0

                    # Paper Eq. 7: vi(t+) = vi(t) + Δv · (1 − vi(t)) · exp(−ni/N)
                    access_log = f.get("access_steps", [])
                    access_log.append(step_idx)
                    ni = sum(1 for s in access_log if step_idx - s <= self.FADE_W)
                    access_log[:] = [s for s in access_log if step_idx - s <= self.FADE_W]
                    v_old = f.get("v_i", 0.0)
                    f["v_i"] = min(1.0, v_old + self.FADE_DELTA_V * (1.0 - v_old) * pow(2.718281828, -ni / self.FADE_N))
                top_facts = sorted(top_facts, key=lambda x: str(x.get("steps", "")))
            else:
                top_facts = []

            if top_facts:
                self._trace("facts_selected_for_prompt", step_idx,
                            selected=[{"fact": f.get("fact",""), "steps": f.get("steps",""),
                                       "v_i": f.get("v_i"), "shield_active": f.get("shield_active", False)}
                                      for f in top_facts],
                            total_active=len(active_facts), total_buffer=len(self._enriched_buffer))

            enriched_lines = []
            if top_facts:
                enriched_lines.append("Enriched memory from previous steps (fade-enriched):")
                for fact in top_facts:
                    span = fact.get("steps", "")
                    text = fact.get("fact", "")
                    enriched_lines.append(f"Steps {span}: \"{text}\"")
            enriched_block = "\n".join(enriched_lines) if enriched_lines else ""

            self.prompt_builder.update_observation(
                text_obs=text_obs,
                mission=mission,
                step_idx=step_idx,
            )
            messages = self.prompt_builder.build_messages(valid_actions)

            if enriched_block:
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        messages[i] = dict(messages[i])
                        messages[i]["content"] = (
                            f"{messages[i]['content']}\n\n--- Enriched memory ---\n{enriched_block}"
                        )
                        break

            response = self.llm.generate_action_structured(
                messages,
                temperature=self.temperature,
                timeout=self.timeout,
                valid_actions=valid_actions,
            )
        elif self.memory_type == "trajectory":
            # Trajectory: full step-by-step history via HistoryPromptBuilder.
            self.prompt_builder.update_observation(
                text_obs=text_obs,
                mission=mission,
                step_idx=step_idx,
            )
            messages = self.prompt_builder.build_messages(valid_actions)
            response = self.llm.generate_action_structured(
                messages,
                temperature=self.temperature,
                timeout=self.timeout,
                valid_actions=valid_actions,
            )
        else:
            raise ValueError(f"Unsupported memory_type: {self.memory_type}")

        action, reason = self.extract_action(response, valid_actions)
        # Optional future: pass reason into prompt history (needs HistoryPromptBuilder.update_reasoning)
        # self.prompt_builder.update_reasoning(reason)
        self.prompt_builder.update_action(action)

        return {
            "action": action,
            "reason": reason
        }
    
    
    def reset(self, seed=None):
        """Reset the prompt builder"""
        if seed is not None:
            self.rng.seed(seed)
        if self.prompt_builder is not None:
            self.prompt_builder.reset()
        self._enriched_buffer.clear()
        self._enrichment_step_buffer.clear()
        self._reflection_step_history.clear()
        self._trace_log.clear()
        if self.memory_type == "fade_enriched_history":
            self._fade_reactivation_count = 0
            self._fade_priming_count = 0
            self._fade_tag_trigger_count = 0
            self._fade_tag_expire_count = 0
            self._fade_contradiction_count = 0
            self._fade_compatible_count = 0
            self._fade_subsume_count = 0
            self._fade_fusion_count = 0
            self._mission_embedding = None
            self._current_obs_embedding = None
            self._last_obs_text = ""

    def start_episode(self, episode_id: int, mission: str, seed=None,
                       system_prompt: str = None, env_name: str = ""):
        self.reset(seed=seed)
        self._env_name = env_name
        if self.memory_type == "fade_enriched_history":
            self._current_mission = mission
            try:
                self._mission_embedding = self.llm.embed(mission)
            except Exception:
                self._mission_embedding = None

        if self.memory_type == "reflection" and env_name and system_prompt:
            reflections = load_reflections(env_name, max_reflections=3)
            if reflections:
                block_lines = ["Previous self-reflections from earlier episodes:"]
                block_lines += [f"- {r}" for r in reflections]
                system_prompt = f"{system_prompt}\n\n" + "\n".join(block_lines)

        if system_prompt is not None and self.prompt_builder is not None:
            self.prompt_builder.update_instruction_prompt(system_prompt)

    def observe_step(self, step_idx, prev_text_obs, action, step_result):
        # For reflection, accumulate step data so end_episode can generate a reflection
        if self.memory_type == "reflection":
            self._reflection_step_history.append({
                "step": step_idx,
                "observation": str(prev_text_obs),
                "action": str(action),
                "mission": step_result.state.mission if hasattr(step_result, "state") else "",
            })

        # For enriched variants, maintain a non-overlapping window of the last k steps
        if self.memory_type in {
            "enriched",
            "enriched_history",
            "fade_enriched_history",
        }:
            obs_text = str(prev_text_obs)
            self._enrichment_step_buffer.append(
                {
                    "step": step_idx,
                    "observation": obs_text,
                    "action": str(action),
                }
            )
            if len(self._enrichment_step_buffer) == self.ENRICHMENT_WINDOW_K:
                window = list(self._enrichment_step_buffer)
                self._enrichment_step_buffer.clear()

                # Build enrichment prompt adapted from your spec
                mission = step_result.state.mission if hasattr(step_result, "state") else ""
                prompt = (
                    "You are a memory encoder in a navigation agent's memory system.\n"
                    "Your task is to transform a window of 3 navigation steps into a "
                    "single compact, self-contained memory fact.\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Information Filtering\n"
                    "   - Discard information irrelevant to the mission.\n"
                    "   - If the steps add no new navigational information, output an empty string.\n"
                    "2. Context Normalization\n"
                    "   - Resolve relative positions into goal-relevant descriptions.\n"
                    "   - Ensure the fact is interpretable WITHOUT access to prior steps.\n"
                    "3. Mission Anchoring\n"
                    "   - Reference the current subgoal or mission explicitly.\n"
                    "4. Fact Extraction\n"
                    "   - Synthesize the 3 steps into ONE minimal, indivisible factual statement.\n\n"
                    f"MISSION: {mission}\n"
                    f"STEP {window[0]['step']} | OBS: {window[0]['observation']} | ACTION: {window[0]['action']}\n"
                    f"STEP {window[1]['step']} | OBS: {window[1]['observation']} | ACTION: {window[1]['action']}\n"
                    f"STEP {window[2]['step']} | OBS: {window[2]['observation']} | ACTION: {window[2]['action']}\n\n"
                    "Output one sentence only. If there is no useful information, output an empty string."
                )

                fact_text = self.llm.generate(
                    messages=[
                        {
                            "role": "system",
                            "content": "You compress navigation trajectories into concise memory facts.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    timeout=self.timeout,
                )
                fact = str(fact_text).strip()
                if fact:
                    span = f"{window[0]['step']}-{window[2]['step']}"
                    self._trace("fact_encoded", step_idx,
                                window=[{"step": w["step"], "action": w["action"]} for w in window],
                                encoded_fact=fact, span=span)
                    entry = {"steps": span, "fact": fact}
                    if self.memory_type == "fade_enriched_history":
                        try:
                            emb = self.llm.embed(fact)
                        except Exception:
                            emb = None
                        entry.update(
                            {
                                "v_i": 1.0,
                                "age_i": 0,
                                "f_i": 0,
                                "access_steps": [],
                                "embedding": emb,
                                "creation_step": step_idx,
                                "layer_i": "SML",
                                "state": "active",
                                "reactivation_hits": 0,
                                "superseded_by": "",
                                "shield_active": False,
                                "trigger_condition": "",
                                "trigger_embedding": None,
                                "shield_start_step": 0,
                            }
                        )

                        # Predictive tagging: try rule-based trigger first
                        trigger = self._generate_trigger_rule_based(
                            fact, mission,
                        )
                        if trigger:
                            entry["shield_active"] = True
                            entry["trigger_condition"] = trigger
                            entry["shield_start_step"] = step_idx
                            if self._fade_enriched_history_cfg.get("trigger_match", "text") == "embedding":
                                try:
                                    entry["trigger_embedding"] = self.llm.embed(trigger)
                                except Exception:
                                    entry["shield_active"] = False
                            else:
                                entry["trigger_embedding"] = None
                            self._trace("shield_created_rule", step_idx,
                                        fact=fact, trigger_condition=trigger)

                        self._resolve_conflicts(entry, step_idx)
                    self._enriched_buffer.append(entry)
                    if len(self._enriched_buffer) > self.MAX_ENRICHED_FACTS * 2:
                        buf_before = len(self._enriched_buffer)
                        if self.memory_type == "fade_enriched_history":
                            active = [f for f in self._enriched_buffer if f.get("state") == "active"]
                            dormant = [f for f in self._enriched_buffer if f.get("state") == "dormant"]
                            dormant.sort(key=lambda f: f.get("v_i", 0), reverse=True)
                            dormant = dormant[:self.MAX_ENRICHED_FACTS]
                            active = active[-self.MAX_ENRICHED_FACTS:]
                            self._enriched_buffer = active + dormant
                        else:
                            old_buffer = [
                                f for f in self._enriched_buffer
                                if f.get("state", "active") == "active"
                            ]
                            kept = old_buffer[-self.MAX_ENRICHED_FACTS:]
                            dropped = old_buffer[:-self.MAX_ENRICHED_FACTS] if len(old_buffer) > self.MAX_ENRICHED_FACTS else []
                            if dropped:
                                self._trace("fact_dropped_fifo", step_idx,
                                            dropped_facts=[f.get("fact", "") for f in dropped],
                                            buffer_size_before=buf_before,
                                            buffer_size_after=len(kept))
                            self._enriched_buffer = kept

        # For fade_enriched_history, apply decay / promotion / dormancy each step
        if self.memory_type == "fade_enriched_history" and self._enriched_buffer:
            mission = self._current_mission
            updated_buffer = []
            for fact in self._enriched_buffer:
                v = float(fact.get("v_i", 1.0))
                age = int(fact.get("age_i", 0)) + 1
                f_count = int(fact.get("f_i", 0))
                layer = str(fact.get("layer_i", "SML"))
                state = str(fact.get("state", "active"))

                # Paper Eq. 2 relevance: prefer embedding cosine, fall back to token overlap
                fact_emb = fact.get("embedding")
                if fact_emb is not None and self._mission_embedding is not None:
                    goal_relevance = max(0.0, self._cosine_similarity(fact_emb, self._mission_embedding))
                else:
                    goal_relevance = self._mission_conditioned_relevance(
                        fact.get("fact", ""), mission,
                    )

                # Paper Eq. 2: I(t) = α·rel(c,Q) + β·f̃/(1+f̃) + γ·recency
                # f̃_i = Σ exp(-κ(t - t_j)) — time-decayed access rate
                recency = pow(2.718281828, -self.FADE_DELTA_RECENCY * age)
                access_steps = fact.get("access_steps", [])
                if access_steps:
                    f_tilde = sum(
                        pow(2.718281828, -self.FADE_KAPPA * max(0, step_idx - s))
                        for s in access_steps
                    )
                else:
                    f_tilde = float(f_count)
                freq_term = f_tilde / (1.0 + f_tilde)
                # Paper Eq. 9: multiplicative importance reduction from compatible redundancy
                importance_factor = float(fact.get("importance_factor", 1.0))
                I = (self.FADE_ALPHA * goal_relevance
                     + self.FADE_BETA * freq_term
                     + self.FADE_GAMMA * recency) * importance_factor

                if I >= self.FADE_THETA_PROMOTE:
                    layer = "LML"
                elif I < self.FADE_THETA_DEMOTE:
                    layer = "SML"

                # Paper Eq. 5: λ = λ_base · exp(-μ·I)
                # Paper Eq. 13: fused memories get reduced decay via ξ_fused
                xi = float(fact.get("fusion_decay_factor", 1.0))
                lam = self.FADE_LAMBDA_BASE * xi * pow(2.718281828, -self.FADE_MU * I)

                # Predictive tagging: shielded facts decay at reduced rate
                if fact.get("shield_active", False):
                    shield_age = step_idx - fact.get("shield_start_step", 0)
                    if shield_age > self.FADE_TAG_MAX_SHIELD_STEPS:
                        fact["shield_active"] = False
                        self._fade_tag_expire_count += 1
                    else:
                        lam = min(lam, self.FADE_TAG_DECAY_SHIELD)

                alpha = 0.8 if layer == "LML" else 1.2
                v = v * pow(2.718281828, -lam * (age ** alpha))

                v_before = float(fact.get("v_i", 1.0))
                state_before = str(fact.get("state", "active"))

                # Lifecycle: active -> dormant -> dropped
                if state == "active" and v < self.DORMANT_THRESHOLD:
                    state = "dormant"
                elif state == "dormant":
                    if v < self.DROP_THRESHOLD or age > self.MAX_DORMANT_AGE:
                        self._trace("fact_dropped", step_idx,
                                    fact=fact.get("fact", ""), v_i=v, age=age,
                                    reason="below_threshold" if v < self.DROP_THRESHOLD else "max_age")
                        continue

                if self._trace_enabled and (state != state_before or abs(v - v_before) > 0.05):
                    self._trace("fact_decayed", step_idx,
                                fact=fact.get("fact", ""), v_i_before=v_before,
                                v_i_after=v, state_before=state_before, state_after=state,
                                layer=layer, shield_active=fact.get("shield_active", False))

                fact["v_i"] = v
                fact["age_i"] = age
                fact["layer_i"] = layer
                fact["state"] = state
                updated_buffer.append(fact)

            self._enriched_buffer = updated_buffer

            # Paper §2.4: periodically attempt memory fusion
            if step_idx > 0 and step_idx % self.FADE_FUSION_INTERVAL == 0:
                self._attempt_memory_fusion(step_idx)

            # Predictive tagging: batch LLM call for unshielded facts
            if step_idx > 0 and step_idx % self.FADE_TAG_BATCH_INTERVAL == 0:
                self._generate_triggers_batch_llm(step_idx)

        return None

    def end_episode(self, total_reward: float, terminated: bool):
        if self.memory_type != "reflection" or not self._env_name:
            return None

        recent = self._reflection_step_history[-self.MAX_REFLECTION_STEPS_FOR_SUMMARY:]
        traj_lines = []
        for s in recent:
            obs = str(s.get("observation", "")).replace("\n", " ")
            act = str(s.get("action", ""))
            traj_lines.append(f"Step {s.get('step', 0)} | Obs: \"{obs}\" | Action: \"{act}\"")
        traj_text = "\n".join(traj_lines)

        mission = recent[-1].get("mission", "") if recent else ""
        outcome = "SUCCESS" if terminated else "FAILURE"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert BabyAI agent reflecting on a completed episode. "
                    "Given the mission, outcome, and recent trajectory, write a concise "
                    "self-reflection that will help improve performance in future episodes."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Mission: {mission}\n"
                    f"Outcome: {outcome}\n\n"
                    f"Recent trajectory (last {len(recent)} steps):\n{traj_text}\n"
                ),
            },
        ]

        try:
            reflection_struct = self.llm.generate_reflection(
                messages, temperature=0.3, timeout=30
            )
            summary = str(reflection_struct.get("summary", "")).strip()
            strategy = str(reflection_struct.get("strategy", "")).strip()
            lessons = reflection_struct.get("lessons", []) or []
            lessons_text = "; ".join(str(x) for x in lessons)

            pieces = [p for p in [summary, strategy, lessons_text] if p]
            reflection_text = " ".join(pieces)
            if reflection_text:
                append_reflection(self._env_name, reflection_text, max_reflections=3)
        except Exception:
            pass

        return None