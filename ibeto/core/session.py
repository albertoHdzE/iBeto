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

    def stream(self, user_text: str, image: str | None = None) -> Iterator[str]:
        """Stream the assistant reply for one turn, updating history in place.

        `image` is an optional data URL. It is sent to the model for this turn
        only, then dropped from stored history so old pixels don't consume the
        (small) local context window on later turns.
        """
        if image:
            content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image}},
            ]
        else:
            content = user_text
        self.messages.append({"role": "user", "content": content})

        chunks: list[str] = []
        for delta in self.backend.stream(self.messages):
            chunks.append(delta)
            yield delta

        if image:
            self.messages[-1]["content"] = f"{user_text} [showed an image]"
        self.messages.append({"role": "assistant", "content": "".join(chunks)})
