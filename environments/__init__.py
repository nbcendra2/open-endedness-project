"""
Functionalities:

    Public entry point for creating wrapped BabyAI environments

Usage in the rest of the project:

    from environments import make_env
    env = make_env(env_name="BabyAI-MixedTrainLocal-v0/goto")
    state = env.reset(seed=42)
    result = env.step("go forward")

make_env returns an EnvWrapper (see env_wrapper.py) which handles Gym setup,
text wrapping, action validation, and normalised dataclasses (EnvState, StepResult).
"""

from environments.env_wrapper import EnvState, EnvWrapper, StepResult


def make_env(
    env_name="BabyAI-MixedTrainLocal-v0",
    gym_kwargs=None,
    invalid_action_mode="fallback",
    fallback_action="go forward",
):
    """
    Create a BabyAI environment wrapped in EnvWrapper

    Args:
        env_name:
            Gymnasium environment ID (e.g. "BabyAI-GoToLocal-v0") or a MixedTrain
            pseudo-id with a "/goal" suffix (e.g. "BabyAI-MixedTrainLocal-v0/goto")
        gym_kwargs:
            Optional kwargs passed into gym.make(env_name, **gym_kwargs)
            Use this to configure environment-specific options.
        invalid_action_mode:
            How to handle invalid LLM actions:
              - "strict": raise ValueError
              - "fallback": replace with fallback_action and continue
        fallback_action:
            Action string to use when invalid_action_mode="fallback" and the proposed
            action is invalid. Should be one of the BabyAI action strings (e.g. "go forward").

    Returns:
        EnvWrapper: a project-level environment adapter
    """

    return EnvWrapper(
        env_name=env_name,
        gym_kwargs=gym_kwargs,
        invalid_action_mode=invalid_action_mode,
        fallback_action=fallback_action,
    )
