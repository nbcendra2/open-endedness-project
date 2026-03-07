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