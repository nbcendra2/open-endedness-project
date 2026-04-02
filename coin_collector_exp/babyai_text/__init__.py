from .clean_lang_wrapper import BabyAITextCleanLangWrapper # This wrapper is used to clean the language of the environment and return the prompt for the LLM agent to interact with the environment.

# These are the actions that the agent can take and this is needed for the agent to be able to interact with the environment. 
# LLM takes this as input and returns the action to take.

ACTIONS = {
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object in front of you",
    "drop": "drop the object you are holding",
    "toggle": "interact with the object in front of you, such as opening a door",
}

# This function generates the instruction prompt for the agent.
# It takes the environment and the mission as input and returns the instruction prompt
# for the LLM agent to interact with the environment.

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
- When target orientation is ambiguous, commit to one turn direction for at least 2 steps unless the target becomes directly ahead.

""".strip()

    return instruction_prompt
