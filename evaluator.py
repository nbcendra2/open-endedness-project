import gymnasium as gym
import minigrid
from babyai_text.clean_lang_wrapper import BabyAITextCleanLangWrapper
from agent.random_agent import RandomAgent
from environments import make_env

from typing import Any, Dict, List
import json
import os
from omegaconf import DictConfig, OmegaConf

# minigrid.register_minigrid_envs()

class Evaluator:

    def __init__(self, config):
        self.config = config
        self.env_name = self.config.env.name

        self.env = make_env(
            env_name=self.env_name,
            gym_kwargs=OmegaConf.to_container(self.config.env.gym_kwargs, resolve=True),
            invalid_action_mode=self.config.env.invalid_action_mode,
            fallback_action=self.config.env.fallback_action,
        )
        self.seed = int(self.config.eval.seed)
        self.num_episodes = int(self.config.eval.num_episodes)
        self.max_steps = int(self.config.eval.max_steps_per_episode)
        self.out_json = self.config.eval.out_json

        self.agent = RandomAgent(seed=int(self.seed))

        # placholder for future memory implementation: 270226
        self.memory = None

    def run_episode(self, episode_idx):
        """Run a single episode and return the results."""

        state = self.env.reset(seed = self.seed + episode_idx)
        self.agent.reset(seed=self.seed + episode_idx)

        steps: List[Dict[str, Any]] = []
        total_reward = 0.0
        terminated = False
        truncated = False

        for t in range(self.max_steps):
            out = self.agent.act(
                text_obs=state.text_obs,
                mission=state.mission,
                valid_actions=state.valid_actions,
                step_idx=t,
            )
            proposed_action = str(out.get("action", ""))

            step = self.env.step(proposed_action)
            total_reward += step.reward

            steps.append(
                {
                    "episode": episode_idx,
                    "step_idx": t,
                    "mission": state.mission,
                    "text_obs": state.text_obs,
                    "valid_actions": state.valid_actions,
                    "agent_output": {
                        "action": proposed_action,
                        "reason": out.get("reason", ""),
                    },
                    "action_used": step.action_used,
                    "action_was_valid": step.action_was_valid,
                    "reward": step.reward,
                    "terminated": step.terminated,
                    "truncated": step.truncated,
                    "env_reason": step.reason,
                }
            )
            state = step.state
            terminated = step.terminated
            truncated = step.truncated
            if terminated or truncated:
                break

        return {
            "episode": episode_idx,
            "seed": self.seed + episode_idx,
            "num_steps": len(steps),
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "steps": steps,
        }
    def run(self):
        out_dir = os.path.dirname(self.out_json)

        if out_dir:
            os.makedirs(out_dir,exist_ok=True)
        episodes = [self.run_episode(ep) for ep in range(self.num_episodes)]

        result = {
            "env_name": self.env_name,
            "num_episodes": self.num_episodes,
            "max_steps": self.max_steps,
            "episodes": episodes,
        }

        with open(self.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def close(self) -> None:
        self.env.close()

if __name__ == "__main__":
    cfg = OmegaConf.load("config/config.yaml")
    evaluator = Evaluator(cfg)
    evaluator.run()