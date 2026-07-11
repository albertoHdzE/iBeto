"""Smoke tests that run without a live LM Studio server."""

from ibeto.config import Config, load_config
from ibeto.core.session import ConversationSession
from ibeto.llm.lmstudio import LMStudioBackend
from ibeto.memory import load_history, save_history
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


def test_history_roundtrip_and_resume(tmp_path):
    path = tmp_path / "chat_history.json"
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    save_history(messages, path)

    loaded = load_history(path)
    assert loaded == messages[1:]  # system prompt excluded

    resumed = ConversationSession(object(), system_prompt="new-sys", history=loaded)
    assert resumed.messages[0] == {"role": "system", "content": "new-sys"}
    assert resumed.messages[1:] == loaded


def test_load_history_missing_returns_empty(tmp_path):
    assert load_history(tmp_path / "nope.json") == []


def test_tts_empty_text_is_noop():
    # speak("") must return without spawning `say`.
    from ibeto.audio.tts import speak

    speak("")  # should not raise

