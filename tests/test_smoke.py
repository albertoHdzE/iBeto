"""Smoke tests that run without a live LM Studio server."""

from ibeto.config import Config, load_config
from ibeto.core.session import ConversationSession
from ibeto.llm.lmstudio import LMStudioBackend
from ibeto.prompts import load_prompt


def test_backend_importable():
    assert LMStudioBackend is not None


def test_config_defaults_load():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.base_url.startswith("http")
    assert cfg.model


def test_assistant_prompt_loads():
    text = load_prompt("assistant")
    assert "iBeto" in text


def test_session_records_history():
    class FakeBackend:
        def stream(self, messages):
            yield "Hi"
            yield " there"

    session = ConversationSession(FakeBackend(), system_prompt="sys")
    reply = "".join(session.stream("hello"))

    assert reply == "Hi there"
    assert session.messages[0] == {"role": "system", "content": "sys"}
    assert session.messages[-2] == {"role": "user", "content": "hello"}
    assert session.messages[-1] == {"role": "assistant", "content": "Hi there"}
