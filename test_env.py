"""Functionality: Smoke test that BabyAI registers and a single env step runs

No agent or LLM; use this to verify gymnasium + minigrid install after setup
"""

import gymnasium as gym

# Registers BabyAI env ids with gymnasium; without this import gym.make may fail
import minigrid

# Minimal run: one env, one reset, no wrappers or fixed seed
env = gym.make("BabyAI-GoToLocal-v0")
_obs, info = env.reset()
print("Environment works!")
print(f"Mission: {info.get('mission', 'N/A')}")
env.close()
