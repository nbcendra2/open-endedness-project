from unittest import result

import gymnasium as gym
import minigrid
from babyai_text.clean_lang_wrapper import BabyAITextCleanLangWrapper
from agent.random_agent import RandomAgent
from agent.base import BaseAgent
from environments import make_env

from typing import Any, Dict, List
import json
import os
from omegaconf import DictConfig, OmegaConf

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from agent import build_agent
from agent.memory_agent import MemoryAgent

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

        self.agent_type = self.config.agent.type
        self.agent = None
        self.memory = None
        # openai model parameters
        self.model_name = self.config.openai_model.name
        self.temperature = self.config.openai_model.temperature
        self.timeout = self.config.openai_model.timeout

    def run_episode(self, episode_idx):
        """Run a single episode and return the results."""
        state = self.env.reset(seed = self.seed + episode_idx)
        system_prompt = self.env.get_instruction_prompt(state.mission)

        # initialize agent based on agent_type in config
        self.agent = self.build_agent(config = self.config, system_prompt=system_prompt)
        self.agent.start_episode(episode_id=episode_idx, mission=state.mission, seed=self.seed + episode_idx)

        steps: List[Dict[str, Any]] = []
        total_reward = 0.0
        terminated = False
        truncated = False

        prev_step_result = None

        for t in range(self.max_steps):
            out = self.agent.act(
                text_obs=state.text_obs,
                mission=state.mission,
                valid_actions=state.valid_actions,
                step_idx=t,
            )

            proposed_action = str(out.get("action", ""))

            step = self.env.step(proposed_action)

            # if memory exist record the step in memory
            self.agent.observe_step(step_idx=t, prev_text_obs=state.text_obs,
                    action=proposed_action, step_result=step)
            prev_step_result = step
            total_reward += step.reward

            # add memory rewriting here in the future: 270226
            # self.agent.observe(step) # add memory to JSON

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
        self.agent.end_episode(total_reward=total_reward, terminated=terminated)
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

        episodes = []
        for ep in tqdm(range(self.num_episodes), total=self.num_episodes, desc="Episodes"):
            episodes.append(self.run_episode(ep))

        result = {
            "env_name": self.env_name,
            "num_episodes": self.num_episodes,
            "max_steps": self.max_steps,
            "episodes": episodes,
        }
        p = Path(self.out_json)  # e.g. runs/rollout.json
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = p.with_name(f"{p.stem}_{self.agent_type}_{ts}{p.suffix}")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def close(self) -> None:
        self.env.close()

if __name__ == "__main__":
    cfg = OmegaConf.load("config/config.yaml")
    evaluator = Evaluator(cfg)
    evaluator.run()