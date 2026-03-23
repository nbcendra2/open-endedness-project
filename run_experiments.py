import subprocess
import yaml
import itertools
import os
import glob
from datetime import datetime

# CONFIG_PATH = "config\config.yaml"
CONFIG_PATH = "config/config.yaml"
# Feel free to change these!
envs = ["BabyAI-MixedTrainLocal-v0/goto", "BabyAI-MixedTrainLocal-v0/pickup"]
memory_types = [
    "baseline",
    "trajectory",
    "reflection"
]
seeds = [0]
num_episodes = [5]
max_steps_per_episode = [20]


def update_config(env, memory_type, seed, num_episode, max_step_per_episode):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["env"]["name"] = env
    config["agent"]["params"]["memory_type"] = memory_type
    config["eval"]["seed"] = seed
    config["eval"]["num_episodes"] = num_episode
    config["eval"]["max_steps_per_episode"] = max_step_per_episode

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)


def get_latest_rollout_file():
    files = glob.glob("runs/rollout_*.jsonl")
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def run_command(cmd):
    print(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    os.makedirs("runs", exist_ok= True)

    for env, memory_type, seed, num_episode, max_step_per_episode in itertools.product(envs, memory_types, seeds, num_episodes, max_steps_per_episode):
        update_config(env, memory_type, seed, num_episode, max_step_per_episode)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_command("python evaluator.py")
        rollout_file = get_latest_rollout_file()
        run_command(f"python postprocess_rollouts.py --input-file {rollout_file}")


if __name__ == "__main__":
    main()