from typing import Dict, List


def action_json_schema(valid_actions: List[str]) -> Dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_action",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "action": {"type": "string", "enum": valid_actions},
                },
                "required": ["reason", "action"],
                "additionalProperties": False,
            },
        },
    }


def planning_json_schema() -> Dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string"},
                },
                "required": ["plan"],
                "additionalProperties": False,
            },
        },
    }


def reflection_json_schema() -> Dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "episode_reflection",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "strategy": {"type": "string"},
                    "lessons": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["summary", "strategy", "lessons"],
                "additionalProperties": False,
            },
        },
    }