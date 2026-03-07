import random as rng

class RandomAgent:
    def __init__(self, seed=0):
        self.rng = rng.Random(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)

    def act(self, text_obs, mission, valid_actions, step_idx):
        a = self.rng.choice(valid_actions)
        return {
            "action": a,
            "reason": "random baseline"
        }
    def start_episode(self, episode_id: int, mission: str, seed=None):
        self.reset(seed=seed)

    def observe_step(self, step_idx, prev_text_obs, action, step_result):
        return None

    def end_episode(self, total_reward: float, terminated: bool):
        return None