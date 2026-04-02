import os
import openai
from dotenv import load_dotenv

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