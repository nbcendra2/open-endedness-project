from .clean_lang_wrapper import BabyAITextCleanLangWrapper

# ACTIONS = {
#     "turn left": "turn to the left",
#     "turn right": "turn to the right",
#     "go forward": "take one step forward",
#     "pick up": "pick up the object below you",
#     "drop": "drop the object that you are holding",
#     "toggle": "manipulate the object in front of you",
# }


# def get_instruction_prompt(env, mission="BabyAI-MixedTrainLocal-v0"):
#     action_strings = ",\n".join(f"{action}: {description}" for action, description in ACTIONS.items())

#     instruction_prompt = f"""
# You are an agent playing a simple navigation game. Your goal is to {mission}. The following are the possible actions you can take in the game, followed by a short description of each action:

# {action_strings}.

# In a moment I will present you an observation.

# Tips:
# - Once the desired object you want to interact or pickup in front of you, you can use the 'toggle' action to interact with it.
# - It doesn't make sense to repeat the same action over and over if the observation doesn't change.

# PLAY!
# """.strip()

#     return instruction_prompt

# Then choose exactly one valid action and return JSON only in this format:
# {{
#   "reason": "your reasoning in 10 words",
#   "action": "YOUR CHOSEN ACTION"
# }}
# Do not output anything outside this JSON.

ACTIONS = {
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object in front of you",
    "drop": "drop the object you are holding",
    "toggle": "interact with the object in front of you, such as opening a door",
}

def get_instruction_prompt(env, mission="Nothing"):
    """
    Generate a system prompt for the agent based on the mission and action space.
    """
    action_strings = "\n".join(f'- "{action}": {description}' for action, description in ACTIONS.items())

    instruction_prompt = f"""
You are an agent playing a simple navigation game. Your goal is to {mission}. The following are the possible actions you can take in the game, followed by a short description of each action:
{action_strings}

In a moment I will present you an observation.

Tips:
- You can only interact with an object that is directly in front of you.
- Once the desired object you want to interact or pickup in front of you, you can use the 'toggle' action to interact with it.
- It doesn't make sense to repeat the same action over and over if the observation doesn't change.
- Avoid turn oscillation: if you just turned left, do not immediately turn right on unchanged observations (and vice versa).
- When target orientation is ambiguous, commit to one turn direction for at least 2 steps unless the target becomes directly ahead.

""".strip()

    return instruction_prompt
