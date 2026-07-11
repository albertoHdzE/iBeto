"""LM Studio backend: thin wrapper over its OpenAI-compatible API.

Only communication lives here — no prompt or conversation logic.
"""

from collections.abc import Iterator

from openai import OpenAI


class LMStudioBackend:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model: str | None = None,
        temperature: float = 0.7,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self.temperature = temperature
        # Populated after each stream() so callers can report tokens/sec.
        self.last_usage = None

    def list_models(self):
        return self.client.models.list()

    @property
    def model(self) -> str:
        """Configured model, or the first one LM Studio has loaded."""
        if self._model:
            return self._model
        return self.list_models().data[0].id

    def chat(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield content deltas as they arrive; store token usage on completion."""
        self.last_usage = None
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.usage is not None:
                self.last_usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
