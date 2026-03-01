
from collections import deque
from typing import Optional, List, Dict

class HistoryPromptBuilder:
    """
    Creates a prompt using past observations, actions taken and reasoning done.
    Records previous conversations and outcomes.
    """
    def __init__(
            self, 
            max_text_history: int = 16,
            system_prompt: Optional[str] = None,
            max_cot_history: int = 1,
            ):
        self.max_text_history = max_text_history
        self.system_prompt = system_prompt
        self._events = deque(maxlen=self.max_text_history * 2)  # Stores observations and actions
        self.previous_reasoning = None
        self.max_cot_history = max_cot_history
    

    def update_instruction_prompt(self, instruction: str):
        """Set the system-level instruction prompt."""
        self.system_prompt = instruction

    def update_observation(self, text_obs: dict, mission: str, step_idx: int):
        """Add observation to the prompt history."""
        self._events.append(
            {
                "type": "observation",
                "mission": mission,
                "text_obs": text_obs,
                "step_idx": step_idx,
            })
        
    def update_action(self, action: str):
        """Add an action to the prompt history, including reasoning if available."""
        self._events.append(
            {
                "type": "action",
                "action": action,
                "reasoning": self.previous_reasoning,
            }
        )
    
    def update_reasoning(self, reasoning: str):
        """Set the reasoning text to be included with subsequent actions."""
        self.previous_reasoning = reasoning

    
    def build_messages(self, valid_actions: List[str])-> List[Dict]:
        """Generate a list of Message objects representing the prompt.
        Returns:
            List[Message]: Messages constructed from the event history.
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
Reasoning: {event['reasoning']}
"""
                    }
                )

        return messages        

    def reset(self):
        self._events.clear()
        self.previous_reasoning = None

