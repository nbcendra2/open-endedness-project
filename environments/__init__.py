from environments.env_wrapper import EnvState, EnvWrapper, StepResult


def make_env(
    env_name="BabyAI-MixedTrainLocal-v0",
    gym_kwargs=None,
    invalid_action_mode="fallback",
    fallback_action="go forward",
):
    """
    Create an BabyAI environment wrapped in EnvWrapper.

    Args:
        env_name:
            Gymnasium environment ID (e.g., "BabyAI-MixedTrainLocal-v0").
        gym_kwargs:
            Optional kwargs passed into `gym.make(env_name, **gym_kwargs)`.
            Use this to configure environment-specific options.
        invalid_action_mode:
            How to handle invalid LLM actions:
              - "strict": raise ValueError
              - "fallback": replace with `fallback_action` and continue
        fallback_action:
            Action string to use when `invalid_action_mode="fallback"` and the proposed action is invalid.
            Should be one of the BabyAI action strings (e.g., "go forward").

    Returns:
        EnvWrapper:
            A project-level environment adapter.

    """

    return EnvWrapper(
        env_name=env_name,
        gym_kwargs=gym_kwargs,
        invalid_action_mode=invalid_action_mode,
        fallback_action=fallback_action,
    )


