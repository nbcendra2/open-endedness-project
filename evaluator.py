import gymnasium as gym
import minigrid
from babyai_text.clean_lang_wrapper import BabyAITextCleanLangWrapper
from agent.random_agent import RandomAgent
from environments.env_wrapper import get_descriptions_missions, valid_actions_from_env

import os
import csv

minigrid.register_minigrid_envs()

def run(env_name="BabyAI-MixedTrainLocal-v0", seed=0, num_episodes=3, max_steps=64, out_csv="runs/rollout.csv"):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    base_env = gym.make(env_name, num_dists=0)
    env = BabyAITextCleanLangWrapper(base_env)

    agent = RandomAgent(seed=seed)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode","step","mission","text_obs","action","reward","terminated","truncated","reason"])

        for ep in range(num_episodes):
            obs, info = env.reset(seed=seed+ep)
            agent.reset(seed=seed+ep)

            # mission: safest from wrapper stats
            mission = get_descriptions_missions(obs,info)[0]

            for t in range(max_steps):
                text_obs = get_descriptions_missions(obs,info)[1]
                valid_actions = env.language_action_space

                out = agent.act(text_obs=text_obs, mission=mission, valid_actions=valid_actions, step_idx=t)
                action = out["action"]

                obs, reward, terminated, truncated, info = env.step(action)

                w.writerow([ep, t, mission, text_obs, action, float(reward), terminated, truncated, out.get("reason","")])

                if terminated or truncated:
                    break

            print(f"ep={ep} progression={env.get_stats().get('progression')}")

    env.close()
    print(f"saved: {out_csv}")

if __name__:
    run()