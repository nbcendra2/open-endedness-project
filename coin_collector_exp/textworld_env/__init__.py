"""TextWorld environment utilities — instruction prompts for Coin Collector."""


def get_instruction_prompt(mission: str = "Find and take the coin.") -> str:
    """Build the system prompt for the LLM agent in TextWorld Coin Collector."""
    return (
        f"You are an agent exploring a text-based world.\n"
        f"Your goal: {mission}\n\n"
        f"At each step you receive a text observation describing your surroundings\n"
        f"and a list of valid actions you can take.\n"
        f"You must choose exactly one action from the valid actions list.\n\n"
        f"Strategy tips:\n"
        f"- Use directional commands (go north, go south, go east, go west) to move.\n"
        f"- Open doors that block your path.\n"
        f"- When you find the coin, use 'take coin' to pick it up and win.\n"
        f"- Explore systematically — remember which rooms you have visited.\n"
        f"- If you seem stuck, try 'look' to re-read your surroundings.\n"
    )