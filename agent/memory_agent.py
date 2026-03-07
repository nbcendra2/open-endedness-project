from agent.base import BaseAgent
from memory import MemoryManager

class MemoryAgent(BaseAgent):
    def __init__(self, model, seed, temperature, timeout, system_prompt, memory_path, retrieval_top_k=3):
        super().__init__(
            model=model,
            seed=seed,
            temperature=temperature,
            timeout=timeout,
            system_prompt=system_prompt,
        )
        self.memory = MemoryManager(
            episodic_path=memory_path,
            retrieval_top_k=retrieval_top_k,
        )
        self._episode_id = None
        self._mission = None

    def start_episode(self, episode_id: int, mission: str, seed=None):
        self._episode_id = episode_id
        self._mission = mission
        super().start_episode(episode_id=episode_id, mission=mission, seed=seed)
        self.memory.start_episode(episode_id=episode_id, mission=mission)

    def act(self, text_obs, mission, valid_actions, step_idx):
        """
        Return action based on updated prompt with retrieved memory.
        """
        mem_text = self.memory.retrieve_as_text(mission=mission, text_obs=text_obs)
        obs_with_mem = text_obs if not mem_text else f"{text_obs}\n\n{mem_text}"
        return super().act(
            text_obs=obs_with_mem,
            mission=mission,
            valid_actions=valid_actions,
            step_idx=step_idx,
        )

    def observe_step(self, step_idx, prev_text_obs, action, step_result):
        """
        Record the step in the memory.
        """
        self.memory.record_step(
            step_idx=step_idx,
            text_obs=prev_text_obs,
            action=action,
            reward=step_result.reward,
            terminated=step_result.terminated,
            truncated=step_result.truncated,
            action_was_valid=step_result.action_was_valid,
            env_reason=step_result.reason,
        )

    def end_episode(self, total_reward: float, terminated: bool):
        self.memory.finish_episode(total_reward=total_reward,
            success=bool(terminated),
        )
