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


def test_sentence_speaker_splits_stream_into_sentences():
    from ibeto.audio.tts import SentenceSpeaker

    spoken: list[str] = []

    class FakeTTS:
        def speak(self, text):
            spoken.append(text)

        def close(self):
            pass

    spk = SentenceSpeaker(FakeTTS())
    # Simulate token-by-token streaming with sentences split across deltas.
    for delta in ["Hello", " Alberto.", " How are", " you today?", " Bye"]:
        spk.feed(delta)
    spk.finish()  # flushes the trailing "Bye" (no terminal punctuation)
    spk.close()

    assert spoken == ["Hello Alberto.", "How are you today?", "Bye"]


def test_clean_for_speech_strips_markdown_and_emoji():
    from ibeto.audio.tts import clean_for_speech

    assert clean_for_speech("**お元気ですか？**") == "お元気ですか？"
    assert clean_for_speech("### How to say it:") == "How to say it:"
    assert clean_for_speech("*   **Genki:** it is *nice*") == "Genki: it is nice"
    assert clean_for_speech("Hello Alberto! 😊") == "Hello Alberto!"
    assert clean_for_speech("---") == ""          # a rule line becomes nothing
    assert clean_for_speech("see [the docs](http://x.com)") == "see the docs"


def test_split_by_script_separates_language_runs():
    from ibeto.audio.tts import detect_lang, split_by_script

    assert split_by_script("Hello Alberto") == ["Hello Alberto"]
    # English + Japanese in one sentence -> two runs, voiced separately
    runs = split_by_script("In Japanese you say お元気ですか?")
    assert runs == ["In Japanese you say ", "お元気ですか?"]
    assert [detect_lang(r) for r in runs] == ["default", "ja"]
    # kanji stays with its kana run (not mis-split as Chinese)
    assert split_by_script("それは 元気 です") == ["それは 元気 です"]
    # Arabic embedded in English
    runs = split_by_script("It means مرحبا in Arabic")
    assert [detect_lang(r) for r in runs] == ["default", "ar", "default"]


def test_sentence_speaker_drops_symbol_only_chunks():
    from ibeto.audio.tts import SentenceSpeaker

    spoken: list[str] = []

    class FakeTTS:
        def speak(self, text):
            spoken.append(text)

        def close(self):
            pass

    spk = SentenceSpeaker(FakeTTS())
    spk.feed("**Bold.** ")   # markdown emphasis around a sentence
    spk.feed("---\n")        # a horizontal rule -> nothing to say
    spk.feed("Plain text.")
    spk.finish()
    spk.close()

    assert spoken == ["Bold.", "Plain text."]


def test_strip_pronunciation_drops_romaji_from_speech_only():
    from ibeto.audio.tts import strip_pronunciation

    # Romanization in parens right after native script -> dropped (not spoken)
    assert strip_pronunciation("こんにちは (konnichiwa)") == "こんにちは"
    assert strip_pronunciation("You say おはよう (ohayou) to greet") == "You say おはよう to greet"
    assert strip_pronunciation("元気ですか？ (genki desu ka)") == "元気ですか？"
    assert strip_pronunciation("مرحبا (marhaban)") == "مرحبا"
    # A normal English parenthetical (not after native script) is kept
    assert strip_pronunciation("the food (very tasty)") == "the food (very tasty)"


def test_detect_lang_routes_by_script():
    from ibeto.audio.tts import detect_lang

    assert detect_lang("Hello Alberto, how are you?") == "default"
    assert detect_lang("¿Qué tal? Muy bien.") == "default"       # Latin -> default
    assert detect_lang("مرحبا يا ألبرتو") == "ar"                 # Arabic block
    assert detect_lang("你好，阿尔贝托") == "zh"                    # Han, no kana
    assert detect_lang("こんにちは、日本語です") == "ja"            # kana present -> ja
    assert detect_lang("Meet 你 at 3pm") == "zh"                  # dominant non-Latin


def test_sentence_speaker_handles_multibyte_and_newlines():
    from ibeto.audio.tts import SentenceSpeaker

    spoken: list[str] = []

    class FakeTTS:
        def speak(self, text):
            spoken.append(text)

        def close(self):
            pass

    spk = SentenceSpeaker(FakeTTS())
    spk.feed("¿Qué tal?\n")   # Spanish + newline boundary
    spk.feed("わからない。")    # Japanese sentence-final 。
    spk.finish()
    spk.close()

    assert spoken == ["¿Qué tal?", "わからない。"]


def test_cjk_paragraph_splits_on_full_width_periods():
    # Regression: CJK has no space after 。 so the old splitter kept the whole
    # paragraph as one chunk, overflowing Kokoro's phoneme limit (loop/crash).
    from ibeto.audio.tts import SentenceSpeaker

    spoken: list[str] = []

    class FakeTTS:
        def speak(self, text):
            spoken.append(text)

        def close(self):
            pass

    para = "はい、日本語を話せます。漢字も使えますよ。何か手伝いましょうか。"
    spk = SentenceSpeaker(FakeTTS())
    spk.feed(para)
    spk.finish()
    spk.close()

    assert spoken == ["はい、日本語を話せます。", "漢字も使えますよ。", "何か手伝いましょうか。"]


def test_runon_without_boundary_flushes_at_length_cap():
    from ibeto.audio.tts import SentenceSpeaker, _MAX_LATIN

    spoken: list[str] = []

    class FakeTTS:
        def speak(self, text):
            spoken.append(text)

        def close(self):
            pass

    spk = SentenceSpeaker(FakeTTS())
    spk.feed("a" * (_MAX_LATIN + 20))  # no sentence boundary at all
    spk.finish()
    spk.close()

    assert len(spoken[0]) == _MAX_LATIN  # capped so the neural model never overflows
    assert "".join(spoken) == "a" * (_MAX_LATIN + 20)


def test_parse_lang_spec():
    from ibeto.cli import _parse_lang_spec

    assert _parse_lang_spec("all") == ("", None)     # auto-detect
    assert _parse_lang_spec("auto") == ("", None)
    assert _parse_lang_spec("de") == ("de", None)
    assert _parse_lang_spec("german") == ("de", None)
    assert _parse_lang_spec("ge") == ("de", None)
    assert _parse_lang_spec("FR") == ("fr", None)    # case-insensitive
    assert _parse_lang_spec("fr1") == ("fr", 1)      # language + level
    assert _parse_lang_spec("ja2") == ("ja", 2)
    assert _parse_lang_spec("xx") == ("xx", None)    # unknown passes through


def test_voice_command_switches_language_and_immersion():
    from ibeto.cli import _handle_voice_command

    class FakeSTT:
        language = "X"

    class FakeSession:
        def __init__(self):
            self.messages = [{"role": "system", "content": "BASE"}]

    stt, sess = FakeSTT(), FakeSession()

    msg = _handle_voice_command("/de2", stt, sess, "BASE")
    assert stt.language == "de"
    assert "German" in msg
    assert "IMMERSION" in sess.messages[0]["content"]
    assert "intermediate" in sess.messages[0]["content"]

    _handle_voice_command("/all", stt, sess, "BASE")
    assert stt.language == ""
    assert sess.messages[0]["content"] == "BASE"   # immersion directive removed

    assert "/help" in _handle_voice_command("/help", stt, sess, "BASE")
    assert _handle_voice_command("/model gemma", stt, sess, "BASE") is None  # falls through


def test_route_text_by_script_and_language():
    from ibeto.audio.tts import route_text

    # English + Japanese in one sentence -> separate chunks
    langs = [lang for lang, _ in route_text("In Japanese you say お元気ですか")]
    assert "ja" in langs
    assert any(lg in ("en", "de", "fr", "es", "it", "pt") for lg in langs)
    # full-sentence Latin languages are told apart natively
    assert route_text("Bonjour, comment ça va aujourd'hui?")[0][0] == "fr"
    assert route_text("你好")[0][0] == "zh"
    assert route_text("مرحبا")[0][0] == "ar"


def test_pipeline_speaker_synthesizes_each_chunk(monkeypatch):
    import pytest

    sd = pytest.importorskip("sounddevice")
    import numpy as np

    class FakeStream:  # stand-in for sd.OutputStream (no real audio device)
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def write(self, data):
            pass

        def stop(self):
            pass

        def abort(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(sd, "OutputStream", FakeStream)

    synthesized: list[tuple[str, str]] = []

    class FakeSynthTTS:  # has synth_lang() -> takes the gapless pipeline path
        def synth_lang(self, text, lang):
            synthesized.append((lang, text))
            return np.zeros(100, dtype=np.float32), 24000

        def close(self):
            pass

    from ibeto.audio.tts import SentenceSpeaker

    spk = SentenceSpeaker(FakeSynthTTS())
    assert spk._pipeline is True
    for d in ["Hello there.", " 你好。"]:
        spk.feed(d)
    spk.finish()
    spk.close()

    texts = [t for _, t in synthesized]
    assert "Hello there." in texts
    assert "你好。" in texts


def test_model_command_list_resolve_and_filter(monkeypatch):
    from ibeto.cli import _model_command
    from ibeto.llm import manager

    class _M:
        def __init__(self, i):
            self.id = i

    class FakeBackend:
        _model = "google/gemma-3-4b"

        @property
        def model(self):
            return self._model

        def list_models(self):
            class R:
                data = [_M("google/gemma-3-4b"), _M("text-embedding-x"), _M("qwen-4b")]

            return R()

    backend = FakeBackend()

    listing = _model_command("", backend)
    assert "google/gemma-3-4b" in listing
    assert "text-embedding-x" not in listing  # embeddings filtered out
    assert "*" in listing  # current model marked

    assert _model_command("nope", backend).startswith("No model")

    # With lms unavailable, switching just sets the id (JIT loads later).
    monkeypatch.setattr(manager, "lms_available", lambda: False)
    msg = _model_command("qwen", backend)
    assert "qwen-4b" in msg
    assert backend._model == "qwen-4b"


def test_frame_to_data_url():
    import numpy as np

    from ibeto.vision.capture import frame_to_data_url

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    url = frame_to_data_url(frame)
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) > len("data:image/jpeg;base64,")


def test_session_strips_image_from_history():
    class FakeBackend:
        def stream(self, messages):
            # The live turn must carry the multimodal content.
            assert isinstance(messages[-1]["content"], list)
            yield "ok"

    session = ConversationSession(FakeBackend(), system_prompt="sys")
    reply = "".join(session.stream("what is this?", image="data:image/jpeg;base64,AAAA"))

    assert reply == "ok"
    assert session.messages[-2] == {
        "role": "user",
        "content": "what is this? [showed an image]",
    }
    assert session.messages[-1] == {"role": "assistant", "content": "ok"}

