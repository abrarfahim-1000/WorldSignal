"""LLM service using Gemini or OpenAI API."""
from typing import Iterator
from app.config import get_settings


class LLMService:
    def __init__(self):
        settings = get_settings()
        if settings.gemini_api_key:
            from google import genai
            self.gemini_client = genai.Client(api_key=settings.gemini_api_key)
            self.backend = "gemini"
        else:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
            self.backend = "openai"
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a complete response."""
        if self.backend == "gemini":
            response = self.gemini_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": self.temperature, "max_output_tokens": max_tokens}
            )
            return response.text or ""
        else:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content or ""
    
    def stream(self, prompt: str, max_tokens: int = 512) -> Iterator[str]:
        """Stream response tokens for real-time UI updates."""
        if self.backend == "gemini":
            response = self.gemini_client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config={"temperature": self.temperature, "max_output_tokens": max_tokens}
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        else:
            stream = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content


_llm_service = None

def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

__all__ = ["LLMService", "get_llm_service"]
