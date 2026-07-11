"""Conversation session: holds dialogue history and drives the backend.

Deliberately independent of the terminal, audio, or any future UI, so the
same engine backs the CLI today and voice/vision later.
"""

from collections.abc import Iterator


class ConversationSession:
    def __init__(self, backend, system_prompt: str, history: list[dict] | None = None):
        self.backend = backend
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if history:
            self.messages.extend(history)

    def stream(self, user_text: str) -> Iterator[str]:
        """Stream the assistant reply for one turn, updating history in place."""
        self.messages.append({"role": "user", "content": user_text})
        chunks: list[str] = []
        for delta in self.backend.stream(self.messages):
            chunks.append(delta)
            yield delta
        self.messages.append({"role": "assistant", "content": "".join(chunks)})
