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
from llm import build_llm_client
from memory.reflection_store import load_reflections, append_reflection

# minigrid.register_minigrid_envs()

class Evaluator:

    def __init__(self, config):
        self.config = config
        self.env_name = self.config.env.name
        env_gym_kwargs = (
            OmegaConf.to_container(self.config.env.gym_kwargs, resolve=True)
            if "gym_kwargs" in self.config.env and self.config.env.gym_kwargs is not None
            else {}
        )

        self.env = make_env(
            env_name=self.env_name,
            gym_kwargs=env_gym_kwargs,
            invalid_action_mode=self.config.env.invalid_action_mode,
            fallback_action=self.config.env.fallback_action,
        )
        self.seed = int(self.config.eval.seed)
        self.num_episodes = int(self.config.eval.num_episodes)
        self.max_steps = int(self.config.eval.max_steps_per_episode)
        self.out_json = self.config.eval.out_json

        # self.agent_type = self.config.agent.name
        # LLM parameters
        self.provider = str(getattr(self.config.llm, "provider", "openai")).lower()
        self.model_name = self.config.llm.name
        self.temperature = self.config.llm.temperature
        self.timeout = self.config.llm.timeout
        # build agent once; system_prompt updated per episode via start_episode
        self.agent = build_agent(config=self.config, system_prompt=None)
        # memory configuration
        self.memory_type = str(
            getattr(getattr(self.config.agent, "params", {}), "memory_type", "baseline")
        ).lower()
        self.agent_type = self.memory_type
        # separate client for reflection calls
        self.reflection_llm = build_llm_client(provider=self.provider, model=self.model_name)

    def run_episode(self, episode_idx):
        """Run a single episode and return the results."""
        state = self.env.reset(seed = self.seed + episode_idx)
        system_prompt = self.env.get_instruction_prompt(state.mission)

        # For reflection memory, prepend previous self-reflections for this env
        if self.memory_type == "reflection":
            reflections = load_reflections(self.env_name, max_reflections=3)
            if reflections:
                reflections_block_lines = [
                    "Previous self-reflections from earlier episodes:"
                ] + [f"- {r}" for r in reflections]
                reflections_block = "\n".join(reflections_block_lines)
                system_prompt = f"{system_prompt}\n\n{reflections_block}"

        self.agent.start_episode(episode_id=episode_idx, mission=state.mission, seed=self.seed + episode_idx, system_prompt=system_prompt)

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

        # After the episode, update reflection memory if enabled
        if self.memory_type == "reflection":
            # Build a compact trajectory summary from the last few steps
            recent_steps = steps[-16:]
            traj_lines = []
            for s in recent_steps:
                obs = str(s.get("text_obs", "")).replace("\n", " ")
                act = str(s.get("agent_output", {}).get("action", ""))
                traj_lines.append(
                    f"Step {s.get('step_idx', 0)} | Obs: \"{obs}\" | Action: \"{act}\""
                )
            traj_text = "\n".join(traj_lines)

            outcome = "SUCCESS" if terminated else "FAILURE"

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert BabyAI agent reflecting on a completed episode. "
                        "Given the mission, outcome, and recent trajectory, write a concise "
                        "self-reflection that will help improve performance in future episodes."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Mission: {state.mission}\n"
                        f"Outcome: {outcome}\n\n"
                        f"Recent trajectory (last {len(recent_steps)} steps):\n{traj_text}\n"
                    ),
                },
            ]

            try:
                reflection_struct = self.reflection_llm.generate_reflection(
                    messages, temperature=0.3, timeout=30
                )
                # Expect keys: summary, strategy, lessons (list)
                summary = str(reflection_struct.get("summary", "")).strip()
                strategy = str(reflection_struct.get("strategy", "")).strip()
                lessons = reflection_struct.get("lessons", []) or []
                lessons_text = "; ".join(str(x) for x in lessons)

                pieces = [p for p in [summary, strategy, lessons_text] if p]
                reflection_text = " ".join(pieces)
                if reflection_text:
                    append_reflection(self.env_name, reflection_text, max_reflections=3)
            except Exception:
                # If reflection generation fails, we simply skip updating memory
                pass

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

        p = Path(self.out_json)
        if p.suffix.lower() != ".jsonl":
            p = p.with_suffix(".jsonl")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = p.with_name(f"{p.stem}_{self.agent_type}_{ts}{p.suffix}")

        with open(out_path, "w", encoding="utf-8") as f:
            for ep in tqdm(range(self.num_episodes), total=self.num_episodes, desc="Episodes"):
                episode_result = self.run_episode(ep)
                line = {
                    "env_name": self.env_name,
                    "agent_type": self.agent_type,
                    "run_seed": self.seed,
                    "max_steps": self.max_steps,
                    "num_episodes": self.num_episodes,
                    **episode_result,
                }
                json.dump(line, f, ensure_ascii=False)
                f.write("\n")

        return {
            "env_name": self.env_name,
            "num_episodes": self.num_episodes,
            "max_steps": self.max_steps,
            "out_path": str(out_path),
            "format": "jsonl",
        }

    def close(self) -> None:
        self.env.close()

if __name__ == "__main__":
    cfg = OmegaConf.load("config/config.yaml")
    evaluator = Evaluator(cfg)
    evaluator.run()
