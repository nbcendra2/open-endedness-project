"""Functionality: Turn agent trajectory into OpenAI-style chat messages for the LLM

Each environment step adds a user message (observation plus valid actions); each
agent reply adds an assistant message (action taken). The LLM sees this as a
multi-turn chat, not a single blob of text
"""

from collections import deque
from typing import Optional, List, Dict


class HistoryPromptBuilder:
    """Accumulates observations and actions, then builds role/content dicts for the API

    Call update_observation after each env step and update_action after each choice.
    build_messages produces the list passed to the LLM client
    """

    def __init__(
        self,
        max_text_history: int = 16,
        system_prompt: Optional[str] = None,
        # max_cot_history: int = 1,  # reserved if we embed reasoning in prompts later
    ):
        self.max_text_history = max_text_history
        self.system_prompt = system_prompt
        # Ring buffer: pairs of observation + action; maxlen caps how far back we send
        self._events = deque(maxlen=self.max_text_history * 2)
        # Optional future: store latest LLM reason string; wire into build_messages when needed
        # self.previous_reasoning = None
        # self.max_cot_history = max_cot_history

    def update_instruction_prompt(self, instruction: str):
        """Replace the system message (task rules, style) sent once at the start"""
        self.system_prompt = instruction

    def update_observation(self, text_obs: dict, mission: str, step_idx: int):
        """Record one env step as a user-side turn (mission, observation, step index)"""
        self._events.append(
            {
                "type": "observation",
                "mission": mission,
                "text_obs": text_obs,
                "step_idx": step_idx,
            }
        )

    def update_action(self, action: str):
        """Record the chosen discrete action as an assistant-side turn"""
        self._events.append(
            {
                "type": "action",
                "action": action,
            }
        )

    # def update_reasoning(self, reasoning: str):
    #     """Store latest reasoning; uncomment and use in build_messages when needed"""
    #     self.previous_reasoning = reasoning

    def build_messages(self, valid_actions: List[str]) -> List[Dict]:
        """Build messages: optional system, then alternating user (obs) and assistant (action)

        valid_actions is repeated on every observation block so the model knows legal moves
        """
        messages = []
        if self.system_prompt:
            messages.append(
                {"role": "system", "content": self.system_prompt}
            )

        for event in self._events:
            if event["type"] == "observation":
                messages.append(
                    {
                        "role": "user",
                        "content": f"""
Step: {event['step_idx']}
Mission: {event['mission']}
Observation: {event['text_obs']}
Valid Actions: {valid_actions}
"""
                    }
                )
            elif event["type"] == "action":
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"""
Action Taken: {event['action']}
"""
                    }
                )

        return messages

    def reset(self):
        """Clear trajectory so a new episode starts clean"""
        self._events.clear()
        # self.previous_reasoning = None  # when update_reasoning / __init__ fields are enabled
