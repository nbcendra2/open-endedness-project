import os
import json
import re
from dotenv import load_dotenv

from llm import build_llm_client
from prompt_builder.history import HistoryPromptBuilder

import random as rng
from collections import deque
from typing import List, Dict, Any

load_dotenv()


class BaseAgent:
    """Base class for LLM-based agents"""

    MAX_ENRICHED_FACTS = 16
    ENRICHMENT_WINDOW_K = 3

    def __init__(
        self,
        model="gpt-4o-mini",
        seed=0,
        temperature=0.2,
        timeout=10,
        system_prompt=None,
        memory_type: str = "baseline",
        provider: str = "openai",
    ):
        self.rng = rng.Random(seed)
        self.llm = build_llm_client(provider=provider, model=model)
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.memory_type = str(memory_type)
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

        # Buffers used for enriched / fade_enriched / semantic_enriched memory
        self._enriched_buffer: List[Dict[str, Any]] = []
        self._enrichment_step_buffer: List[Dict[str, Any]] = []


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
        if self.memory_type == "baseline":
            # Stateless baseline: no explicit history, only system prompt + mission + current observation
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
        elif self.memory_type == "fade_enriched":
            # Fade-enriched: select top-N facts by strength v_i
            facts = [f for f in self._enriched_buffer if "v_i" in f]
            if facts:
                facts_sorted = sorted(facts, key=lambda x: x.get("v_i", 0.0), reverse=True)
                top_facts = facts_sorted[: self.MAX_ENRICHED_FACTS]
                # update usage stats for selected facts
                for f in top_facts:
                    f["f_i"] = f.get("f_i", 0) + 1
                    f["age_i"] = 0
                    f["v_i"] = min(1.0, f.get("v_i", 0.0) + 0.1)
                # for readability, keep chronological order by steps span
                top_facts = sorted(top_facts, key=lambda x: str(x.get("steps", "")))
            else:
                top_facts = []

            enriched_lines = []
            if top_facts:
                enriched_lines.append("Enriched memory from previous steps (fade-enriched):")
                for fact in top_facts:
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
        elif self.memory_type == "fade_enriched_history":
            # Hybrid: trajectory prompt history + fade-enriched memory block
            facts = [f for f in self._enriched_buffer if "v_i" in f]
            if facts:
                facts_sorted = sorted(facts, key=lambda x: x.get("v_i", 0.0), reverse=True)
                top_facts = facts_sorted[: self.MAX_ENRICHED_FACTS]
                # update usage stats for selected facts
                for f in top_facts:
                    f["f_i"] = f.get("f_i", 0) + 1
                    f["age_i"] = 0
                    f["v_i"] = min(1.0, f.get("v_i", 0.0) + 0.1)
                # for readability, keep chronological order by steps span
                top_facts = sorted(top_facts, key=lambda x: str(x.get("steps", "")))
            else:
                top_facts = []

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
        elif self.memory_type == "semantic_enriched":
            # Semantic-gated enriched: same structure as enriched, but buffer only contains gated facts
            facts = self._enriched_buffer[-self.MAX_ENRICHED_FACTS :]
            enriched_lines = []
            if facts:
                enriched_lines.append("Semantic-gated enriched memory from previous steps:")
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
        elif self.memory_type == "semantic_enriched_history":
            # Hybrid: trajectory prompt history + semantic-gated enriched memory block
            facts = self._enriched_buffer[-self.MAX_ENRICHED_FACTS :]
            enriched_lines = []
            if facts:
                enriched_lines.append("Semantic-gated enriched memory from previous steps:")
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
        else:
            # Default: use trajectory-style history via HistoryPromptBuilder
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
        action, reason = self.extract_action(response, valid_actions)
        self.prompt_builder.update_reasoning(reason)
        self.prompt_builder.update_action(action)

        return {
            "action": action,
            "reason": reason
        }
    
    
    def reset(self, seed=None):
        """Reset the prompt builder"""
        if seed is not None:
            self.rng.seed(seed)
        self.prompt_builder.reset()
        self._enriched_buffer.clear()
        self._enrichment_step_buffer.clear()

    def start_episode(self, episode_id: int, mission: str, seed=None, system_prompt: str = None):
        self.reset(seed=seed)
        if system_prompt is not None:
            self.prompt_builder.update_instruction_prompt(system_prompt)

    def observe_step(self, step_idx, prev_text_obs, action, step_result):
        # For enriched variants, maintain a non-overlapping window of the last k steps
        if self.memory_type in {
            "enriched",
            "enriched_history",
            "fade_enriched",
            "fade_enriched_history",
            "semantic_enriched",
            "semantic_enriched_history",
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
                    entry = {"steps": span, "fact": fact}
                    if self.memory_type in {"fade_enriched", "fade_enriched_history"}:
                        # Initialize FadeMem metadata
                        entry.update(
                            {
                                "v_i": 1.0,
                                "age_i": 0,
                                "f_i": 0,
                                "layer_i": "SML",
                            }
                        )
                    # For semantic_enriched, the LLM acts as a semantic gate:
                    # empty string => window is discarded; non-empty => store fact.
                    self._enriched_buffer.append(entry)
                    if len(self._enriched_buffer) > self.MAX_ENRICHED_FACTS:
                        self._enriched_buffer.pop(0)
        # For fade_enriched, apply decay / promotion / pruning each step
        if self.memory_type in {"fade_enriched", "fade_enriched_history"} and self._enriched_buffer:
            updated_buffer = []
            for fact in self._enriched_buffer:
                v = float(fact.get("v_i", 1.0))
                age = int(fact.get("age_i", 0)) + 1
                f_count = int(fact.get("f_i", 0))
                layer = str(fact.get("layer_i", "SML"))

                # simple keyword-based goal relevance
                text = f"{fact.get('fact', '')}".lower()
                goal_relevance = 1.0 if any(
                    kw in text for kw in ["key", "door", "picked up", "toggled", "opened"]
                ) else 0.0

                beta = 0.1
                recency = pow(2.718281828, -beta * age)  # e^{-beta * age}
                freq_term = 1.0 - pow(2.718281828, -float(f_count))
                I = goal_relevance + freq_term + recency

                theta_promote = 0.7
                theta_demote = 0.3
                if I > theta_promote:
                    layer = "LML"
                elif I < theta_demote:
                    layer = "SML"

                base = 0.05
                gamma = 1.0
                lam = base * pow(2.718281828, -gamma * I)

                alpha = 0.8 if layer == "LML" else 1.2
                v = v * pow(2.718281828, -lam * (age ** alpha))

                v_min = 0.1
                if v < v_min:
                    continue

                fact["v_i"] = v
                fact["age_i"] = age
                fact["layer_i"] = layer
                updated_buffer.append(fact)

            self._enriched_buffer = updated_buffer

        return None

    def end_episode(self, total_reward: float, terminated: bool):
        # no-op for non-memory agents
        return None   
