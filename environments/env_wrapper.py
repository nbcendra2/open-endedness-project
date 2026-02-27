

BABYAI_ACTION_SPACE = [
    "turn left",
    "turn right",
    "go forward",
    "pick up",
    "drop",
    "toggle",
]

def get_descriptions_missions(obs, info):
    """
    BabyAI-Text store missions and descriptions in the obs and info dictionary
    """
    descriptions = info.get("descriptions")
    missions = obs.get("mission")

    return missions, descriptions


def valid_actions_from_env(candidate_action):
    valid_action = None
    if candidate_action in BABYAI_ACTION_SPACE:
        valid_action = candidate_action
    else:
        valid_action = make_dummy_action()