"""
Collect episodes and build archive for text memento agent.
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from environments import make_env
from agent.random_agent import RandomAgent


def collect_episodes(config: DictConfig, num_episodes: int = 200):
    """Collect episodes using random agent."""
    print(f"\n📦 Collecting {num_episodes} episodes...")
    
    env = make_env(
        env_name=config.env.name,
        gym_kwargs=OmegaConf.to_container(config.env.gym_kwargs, resolve=True),
        invalid_action_mode=config.env.invalid_action_mode,
        fallback_action=config.env.fallback_action,
    )
    
    agent = RandomAgent(seed=config.eval.seed)
    max_steps = config.eval.max_steps_per_episode
    
    episodes = []
    
    for ep_idx in tqdm(range(num_episodes)):
        state = env.reset(seed=config.eval.seed + ep_idx)
        agent.reset(seed=config.eval.seed + ep_idx)
        
        trajectory = []
        total_reward = 0.0
        
        for t in range(max_steps):
            out = agent.act(
                text_obs=state.text_obs,
                mission=state.mission,
                valid_actions=state.valid_actions,
                step_idx=t,
            )
            
            action = out["action"]
            step = env.step(action)
            
            trajectory.append({
                "text_obs": state.text_obs,
                "action": action
            })
            
            total_reward += step.reward
            state = step.state
            
            if step.terminated or step.truncated:
                break
        
        episodes.append({
            "mission": state.mission,
            "trajectory": trajectory,
            "reward": total_reward,
            "success": total_reward > 0,
            "num_steps": len(trajectory)
        })
    
    env.close()
    
    success_rate = sum(1 for ep in episodes if ep["success"]) / len(episodes)
    mean_reward = np.mean([ep["reward"] for ep in episodes])
    
    print(f"✅ Collected {len(episodes)} episodes")
    print(f"   Success rate: {success_rate:.1%}")
    print(f"   Mean reward: {mean_reward:.3f}")
    
    return episodes


def build_archive(episodes: list, success_threshold: float = 0.8):
    """Build archive from successful episodes."""
    print(f"\n🏗️  Building archive...")
    
    # Filter successful episodes
    successful = [ep for ep in episodes if ep["success"]]
    
    # Sort by reward (descending) and take top cases
    successful.sort(key=lambda x: x["reward"], reverse=True)
    
    # Take top 50 or top 80% whichever is smaller
    num_cases = min(50, int(len(successful) * success_threshold))
    archive_cases = successful[:num_cases]
    
    # Add embeddings
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    for case in archive_cases:
        mission_embedding = encoder.encode(case["mission"], convert_to_numpy=True)
        case["embedding"] = mission_embedding.tolist()
    
    success_rate = sum(1 for c in archive_cases if c["success"]) / len(archive_cases) if archive_cases else 0
    
    print(f"✅ Archive: {len(archive_cases)} cases, {success_rate:.1%} success")
    
    return archive_cases


def main():
    """Main script to collect and save archive."""
    config = OmegaConf.load("config/config.yaml")
    
    # Collect episodes
    episodes = collect_episodes(config, num_episodes=200)
    
    # Build archive
    archive_cases = build_archive(episodes)
    
    # Save archive
    archive_path = config.memory.path
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    
    with open(archive_path, 'w') as f:
        json.dump({"cases": archive_cases}, f, indent=2)
    
    print(f"\n💾 Saved archive to: {archive_path}")


if __name__ == "__main__":
    main()