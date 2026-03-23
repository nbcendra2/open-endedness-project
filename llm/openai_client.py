import os
import json
import openai
from dotenv import load_dotenv

from llm.base_client import BaseLLMClient
from llm.json_output_structure import (
    action_json_schema,
    planning_json_schema,
    reflection_json_schema,
)

load_dotenv()


class OpenAIClient(BaseLLMClient):
    """OpenAI provider — uses native structured-output (response_format)."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(self, messages, temperature=0.2, timeout=10):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
        )
        return response.choices[0].message.content

    def generate_action_structured(self, messages, valid_actions, temperature=0.2, timeout=10):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            response_format=action_json_schema(valid_actions),
        )
        return json.loads(response.choices[0].message.content)

    def generate_planning_structured(self, messages, temperature=0.3, timeout=15):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            response_format=planning_json_schema(),
        )
        return json.loads(response.choices[0].message.content)

    def generate_reflection(self, messages, temperature=0.3, timeout=30):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            response_format=reflection_json_schema(),
        )
        return json.loads(response.choices[0].message.content)
