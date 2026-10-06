from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Generator:
    """Generates a final answer using OpenAI GPT-4o and retrieved context."""

    def __init__(self, model: str = "gemini-3.8-flash"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, question: str, context: List[str]) -> str:
        joined_context = "\n\n".join(context)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer using only the provided context when possible.",
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{joined_context}",
                },
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content
