# test_env.py
import gymnasium as gym
import minigrid  # <-- MUST IMPORT THIS!

# Try without any wrappers or seeds
env = gym.make("BabyAI-GoToLocal-v0")
obs, info = env.reset()
print("✅ Environment works!")
print(f"Mission: {info.get('mission', 'N/A')}")
env.close()

# Try without any wrappers or seeds
env = gym.make("BabyAI-GoToLocal-v0")
obs, info = env.reset()
print("✅ Environment works!")
print(f"Mission: {info.get('mission', 'N/A')}")
env.close()