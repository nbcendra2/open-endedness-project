import os
import json
import openai
from dotenv import load_dotenv

from llm.base_client import BaseLLMClient

load_dotenv()

# DeepSeek supports {"type": "json_object"} but NOT strict JSON schema with
# enum validation.  We inject the schema into the prompt and validate after.

_ACTION_SCHEMA_HINT = (
    '\n\nYou MUST respond with a JSON object matching this exact schema:\n'
    '{{"reason": "<your reasoning in 10 words>", "action": "<one of: {actions}>"}}\n'
    'Do not include any text outside the JSON object.'
)

_PLANNING_SCHEMA_HINT = (
    '\n\nYou MUST respond with a JSON object matching this exact schema:\n'
    '{"plan": "<your revised plan>"}\n'
    'Do not include any text outside the JSON object.'
)

_REFLECTION_SCHEMA_HINT = (
    '\n\nYou MUST respond with a JSON object matching this exact schema:\n'
    '{"summary": "<1-2 sentence recap>", "strategy": "<approach used>", '
    '"lessons": ["<lesson1>", "<lesson2>", ...]}\n'
    'Do not include any text outside the JSON object.'
)

_JSON_MODE = {"type": "json_object"}


class DeepSeekClient(BaseLLMClient):
    """DeepSeek provider — OpenAI-compatible API with prompt-based schema enforcement."""

    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
        )

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _inject_hint(messages: list, hint: str) -> list:
        """Append *hint* to the last user message (non-destructive copy)."""
        messages = [dict(m) for m in messages]
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i]["content"] = messages[i]["content"] + hint
                break
        return messages

    @staticmethod
    def _safe_parse(content: str) -> dict:
        """Parse JSON, stripping markdown fences if present."""
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(text)

    # ── interface ─────────────────────────────────────────────────

    def generate(self, messages, temperature=0.2, timeout=10):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
        )
        return response.choices[0].message.content

    def generate_action_structured(self, messages, valid_actions, temperature=0.2, timeout=10):
        hint = _ACTION_SCHEMA_HINT.format(actions=", ".join(valid_actions))
        messages = self._inject_hint(messages, hint)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            response_format=_JSON_MODE,
        )
        result = self._safe_parse(response.choices[0].message.content)
        # validate action is in valid set
        if result.get("action") not in valid_actions:
            result["action"] = valid_actions[0]
        return result

    def generate_planning_structured(self, messages, temperature=0.3, timeout=15):
        messages = self._inject_hint(messages, _PLANNING_SCHEMA_HINT)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            response_format=_JSON_MODE,
        )
        return self._safe_parse(response.choices[0].message.content)

    def generate_reflection(self, messages, temperature=0.3, timeout=30):
        messages = self._inject_hint(messages, _REFLECTION_SCHEMA_HINT)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            response_format=_JSON_MODE,
        )
        return self._safe_parse(response.choices[0].message.content)
