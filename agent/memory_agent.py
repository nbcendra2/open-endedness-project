import json

from agent.base import BaseAgent
from memory import MemoryManager

_STUCK_ACTIONS = {"turn left", "turn right", "toggle"}

# PLANNING_SYSTEM_PROMPT = (
#     "You are a planning agent. Given a mission, recent trajectory, and memory context, "
#     "produce a short revised plan (1-3 sentences) for how to accomplish the mission. "
#     "Be concrete and actionable. Focus on what to do next, not what went wrong."
# )
PLANNING_SYSTEM_PROMPT = (
    "You are a planning agent controlling a robot in a partially observable grid world. "
    "The robot moves on a grid with discrete actions:\n"
    '- "turn left" / "turn right": rotate 90° in place\n'
    '- "go forward": move one cell forward\n'
    '- "pick up": pick up the object directly in front of you\n'
    '- "toggle": interact with the object directly in front of you\n\n'
    "Given a mission, recent trajectory, and memory context, produce a single concrete revised plan "
    "(1-2 sentences) using the available actions. Prioritize using locations of previously seen objects."
)

REFLECTION_SYSTEM_PROMPT = (
    "You are a reflective agent. Given an episode trajectory and outcome, "
    "produce a JSON object with:\n"
    '- "summary": 1-2 sentence recap of what happened\n'
    '- "strategy": the approach that was used\n'
    '- "lessons": list of short actionable takeaways (what worked, what to avoid)\n'
    "Be concise and specific. Focus on what would help in future similar missions."
)


class MemoryAgent(BaseAgent):
    def __init__(self, model, seed, temperature, timeout, system_prompt, memory_path,
                 retrieval_top_k=3, retriever="embedding", stuck_window=3,
                 reflection=True, planning=True, provider="openai",
                 memory_type="baseline"):
        super().__init__(
            model=model,
            seed=seed,
            temperature=temperature,
            timeout=timeout,
            system_prompt=system_prompt,
            provider=provider,
            memory_type=memory_type,
        )
        self.memory = MemoryManager(
            episodic_path=memory_path,
            retrieval_top_k=retrieval_top_k,
            retriever=retriever,
        )
        self._episode_id = None
        self._mission = None
        self._stuck_window = stuck_window
        self._reflection_enabled = reflection
        self._planning_enabled = planning

    def start_episode(self, episode_id: int, mission: str, seed=None, system_prompt: str = None):
        self._episode_id = episode_id
        self._mission = mission
        super().start_episode(episode_id=episode_id, mission=mission, seed=seed, system_prompt=system_prompt)
        self.memory.start_episode(episode_id=episode_id, mission=mission)

    def act(self, text_obs, mission, valid_actions, step_idx):
        """
        Return action based on updated prompt with retrieved memory.
        Uses same {reason, action} schema as base agent.
        Memory (including plan) is injected ephemerally into the latest message.
        """
        # Replan if stuck (checked before acting)
        if self._planning_enabled and self._is_stuck():
            self._replan()

        # Record clean observation in history
        self.prompt_builder.update_observation(
            text_obs=text_obs,
            mission=mission,
            step_idx=step_idx,
        )

        # Build messages from clean history
        messages = self.prompt_builder.build_messages(valid_actions)

        # Append memory to ONLY the last user message
        mem_text = self.memory.retrieve_as_text(mission=mission, text_obs=text_obs)
        if mem_text and messages and messages[-1]["role"] == "user":
            messages[-1] = dict(messages[-1])
            messages[-1]["content"] = messages[-1]["content"] + f"\n\n--- Memory ---\n{mem_text}"

        response = self.llm.generate_action_structured(
            messages,
            temperature=self.temperature,
            timeout=self.timeout,
            valid_actions=valid_actions,
        )

        action, reason = self.extract_action(response, valid_actions)

        self.prompt_builder.update_reasoning(reason)
        self.prompt_builder.update_action(action)

        return {
            "action": action,
            "reason": reason,
        }

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
        if self._reflection_enabled:
            reflection = self._reflect(total_reward=total_reward, success=terminated)
        else:
            reflection = {"summary": "", "strategy": "", "lessons": []}
        self.memory.finish_episode(
            total_reward=total_reward,
            success=bool(terminated),
            summary=reflection.get("summary", ""),
            strategy=reflection.get("strategy", ""),
            lessons=reflection.get("lessons", []),
        )

    # ── Stuck detection & replanning ─────────────────────────────

    def _is_stuck(self) -> bool:
        """Return True if last N actions are all turns/toggles with no reward."""
        steps = self.memory.working.steps
        if len(steps) < self._stuck_window:
            return False
        recent = steps[-self._stuck_window:]
        stuck = (
            all(s.action in _STUCK_ACTIONS for s in recent)
            and all(s.reward == 0.0 for s in recent)
        )
        if stuck:
            print(f"[DEBUG stuck] STUCK detected at step {recent[-1].step_idx} | last {self._stuck_window} actions: {[s.action for s in recent]}")
        return stuck

    def _replan(self) -> None:
        """Call LLM to produce a revised plan based on current working memory."""
        steps = self.memory.working.steps
        recent = steps[-self._stuck_window:] if len(steps) >= self._stuck_window else steps

        lines = [f"Mission: {self._mission}"]
        if self.memory.working.plan:
            lines.append(f"Previous plan (not working): {self.memory.working.plan}")

        lines.append(f"Recent steps (stuck):")
        for s in recent:
            lines.append(f"  step {s.step_idx}: obs={s.text_obs} -> {s.action}, reward={s.reward}")

        seen = self.memory.working.format_seen_objects()
        if seen:
            lines.append(seen)

        if self.memory.working.insights:
            lines.append("Past lessons:")
            for ins in self.memory.working.insights:
                lines.append(f"  - {ins}")

        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]
        try:
            result = self.llm.generate_planning_structured(
                messages, temperature=1.0, timeout=self.timeout
            )
            plan = result.get("plan", "")
            print(f"[DEBUG replan] New plan: {repr(plan)}")
            if plan:
                self.memory.working.set_plan(plan)
        except Exception as e:
            print(f"[DEBUG replan] Failed: {e}")

    # ── Reflection ────────────────────────────────────────────────

    def _reflect(self, total_reward: float, success: bool) -> dict:
        """Ask the LLM to reflect on the episode and extract lessons."""
        trajectory_text = self._format_trajectory_for_reflection(
            total_reward=total_reward, success=success
        )
        messages = [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": trajectory_text},
        ]
        try:
            return self.llm.generate_reflection(messages, temperature=0.3, timeout=self.timeout)
        except Exception:
            return {"summary": "", "strategy": "", "lessons": []}

    def _format_trajectory_for_reflection(self, total_reward: float, success: bool) -> str:
        lines = [
            f"Mission: {self._mission}",
            f"Outcome: {'SUCCESS' if success else 'FAILED'}, total_reward={total_reward}",
            f"Steps ({len(self.memory.working.steps)}):",
        ]
        for step in self.memory.working.steps:
            valid_str = "" if step.action_was_valid else " [INVALID]"
            lines.append(
                f"  {step.step_idx}: obs={step.text_obs} -> action={step.action}{valid_str}, reward={step.reward}"
            )
        return "\n".join(lines)
