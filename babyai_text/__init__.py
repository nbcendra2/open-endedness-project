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


Then you must choose exactly one of the given actions and output it strictly in the following format: 
{{\n'
"reason": "your reasoning in 10 words",\n'
"action": "YOUR CHOSEN ACTION"\n'
}}\n'                       
"Replace YOUR CHOSEN ACTION with the chosen action. Do not output anything outside the JSON."
""".strip()

    return instruction_prompt
