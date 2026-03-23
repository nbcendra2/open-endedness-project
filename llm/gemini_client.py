import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

from llm.base_client import BaseLLMClient

load_dotenv()


class GeminiClient(BaseLLMClient):
    """Google Gemini provider — uses native structured-output (response_schema)."""

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        self.model_name = model
        self.api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _convert_messages(messages: list):
        """Split OpenAI-style messages into (system_instruction, contents).

        Gemini uses 'user' / 'model' roles (not 'assistant').
        """
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system_instruction = text
            elif role == "user":
                contents.append({"role": "user", "parts": [text]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [text]})
        return system_instruction, contents

    def _make_model(self, system_instruction: str | None):
        """Create a GenerativeModel with the given system instruction."""
        return genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction,
        )

    @staticmethod
    def _safe_parse(text: str) -> dict:
        """Parse JSON, stripping markdown fences if present."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(text)

    # ── interface ─────────────────────────────────────────────────

    def generate(self, messages, temperature=0.2, timeout=10):
        system_instruction, contents = self._convert_messages(messages)
        model = self._make_model(system_instruction)
        response = model.generate_content(
            contents,
            generation_config=genai.GenerationConfig(temperature=temperature),
            request_options={"timeout": timeout},
        )
        return response.text

    def generate_action_structured(self, messages, valid_actions, temperature=0.2, timeout=10):
        system_instruction, contents = self._convert_messages(messages)
        model = self._make_model(system_instruction)

        schema = {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "action": {"type": "string", "enum": valid_actions},
            },
            "required": ["reason", "action"],
        }

        response = model.generate_content(
            contents,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
            request_options={"timeout": timeout},
        )
        result = self._safe_parse(response.text)
        # fallback validation in case model ignores enum
        if result.get("action") not in valid_actions:
            result["action"] = valid_actions[0]
        return result

    def generate_planning_structured(self, messages, temperature=0.3, timeout=15):
        system_instruction, contents = self._convert_messages(messages)
        model = self._make_model(system_instruction)

        schema = {
            "type": "object",
            "properties": {
                "plan": {"type": "string"},
            },
            "required": ["plan"],
        }

        response = model.generate_content(
            contents,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
            request_options={"timeout": timeout},
        )
        return self._safe_parse(response.text)

    def generate_reflection(self, messages, temperature=0.3, timeout=30):
        system_instruction, contents = self._convert_messages(messages)
        model = self._make_model(system_instruction)

        schema = {
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
        }

        response = model.generate_content(
            contents,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
            request_options={"timeout": timeout},
        )
        return self._safe_parse(response.text)
