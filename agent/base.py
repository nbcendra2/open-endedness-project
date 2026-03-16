import os
import json
import re
import openai
from dotenv import load_dotenv

from llm_client import LLMClient
from prompt_builder.history import HistoryPromptBuilder

import random as rng

load_dotenv()

class BaseAgent:
    """Base class for LLM-based agents"""
    def __init__(self, model = "gpt-4o-mini", seed = 0, temperature = 0.2, timeout = 10, system_prompt = None):
        self.rng = rng.Random(seed)
        self.llm = LLMClient(model=model)
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
        self.prompt_builder.update_observation(
            text_obs=text_obs,
            mission=mission,
            step_idx=step_idx,
        )
        messages = self.prompt_builder.build_messages(valid_actions)
        # use json response format to get structured output
        response = self.llm.generate_action_structured(
            messages,
            temperature=self.temperature,
            timeout=self.timeout,
            valid_actions=valid_actions
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

    def start_episode(self, episode_id: int, mission: str, seed=None, system_prompt: str = None):
        self.reset(seed=seed)
        if system_prompt is not None:
            self.prompt_builder.update_instruction_prompt(system_prompt)

    def observe_step(self, step_idx, prev_text_obs, action, step_result):
        # no-op for non-memory agents
        return None

    def end_episode(self, total_reward: float, terminated: bool):
        # no-op for non-memory agents
        return None   
