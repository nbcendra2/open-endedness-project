import os
import openai
from dotenv import load_dotenv
from llm.json_output_structure import action_json_schema
import json

load_dotenv()


class LLMClient:
    def __init__(self, model="gpt-4o-mini"):
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
            # temperature=temperature,
            timeout=timeout,
            response_format=action_json_schema(valid_actions),
        )
        content = response.choices[0].message.content
        return json.loads(content)