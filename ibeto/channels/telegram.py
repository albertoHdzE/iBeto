"""Telegram front-end for iBeto (`ibeto --telegram`).

Reuses the same brain as the terminal — ConversationSession + Whisper STT +
neural TTS — behind a Telegram bot. Long-polling, so it runs behind your
firewall with no public server: your Mac polls Telegram's cloud, your phone
talks to Telegram's cloud, and they never connect directly.

Text and voice notes both ways. Per-user conversations, the same /de.././all
immersion + level commands, romanization shown-not-spoken.

Setup (never committed):
    export TELEGRAM_BOT_TOKEN='token-from-@BotFather'
    export IBETO_TG_ALLOW='your-numeric-id'   # comma-separated allowlist
Run: ibeto --telegram
"""

import asyncio
import os
import subprocess
import tempfile
import wave

import numpy as np

from ibeto.audio.stt import WhisperSTT
from ibeto.audio.tts import make_tts, render_reply
from ibeto.cli import _LANG_NAMES, _help_text, _mode_directive, _parse_lang_spec
from ibeto.config import load_config
from ibeto.core.session import ConversationSession
from ibeto.llm.lmstudio import LMStudioBackend
from ibeto.prompts import load_prompt


def _allowed_ids() -> set[int]:
    raw = os.environ.get("IBETO_TG_ALLOW", "")
    return {int(x) for x in raw.replace(" ", "").split(",") if x.strip().isdigit()}


def _to_ogg(samples: np.ndarray, sr: int) -> str:
    """Encode a waveform to an OGG/Opus file (a Telegram voice note)."""
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    fd, ogg = tempfile.mkstemp(suffix=".ogg")
    os.close(fd)
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    subprocess.run(["ffmpeg", "-y", "-i", wav, "-c:a", "libopus", "-b:a", "32k", ogg],
                   capture_output=True)
    os.unlink(wav)
    return ogg


class _UserState:
    """Per-Telegram-user conversation + immersion language/level."""

    def __init__(self, backend, base_prompt: str, default_lang: str):
        self.session = ConversationSession(backend, base_prompt)
        self.base_prompt = base_prompt
        self.stt_lang = default_lang  # "" = auto-detect


class TelegramChannel:
    """Holds the shared brain and per-user state; wires the Telegram handlers."""

    def __init__(self):
        self.cfg = load_config()
        self.backend = LMStudioBackend(
            base_url=self.cfg.base_url, model=self.cfg.model,
            temperature=self.cfg.temperature,
            enable_thinking=self.cfg.enable_thinking, max_tokens=self.cfg.max_tokens)
        self.base_prompt = load_prompt(self.cfg.system_prompt)
        print(f"Loading Whisper ({self.cfg.whisper_model})...", flush=True)
        self.stt = WhisperSTT(self.cfg.whisper_model, self.cfg.stt_language,
                              self.cfg.whisper_threads)
        self.tts = make_tts(self.cfg)
        self.users: dict[int, _UserState] = {}

    def _user(self, uid: int) -> _UserState:
        if uid not in self.users:
            self.users[uid] = _UserState(self.backend, self.base_prompt, self.cfg.stt_language)
        return self.users[uid]

    def _switch_language(self, user: _UserState, spec: str) -> str | None:
        """Handle a /command that changes language/immersion. None if not one."""
        code, level = _parse_lang_spec(spec)
        if spec in ("all", "auto") or code in _LANG_NAMES:
            user.stt_lang = code
            user.session.messages[0]["content"] = self.base_prompt + _mode_directive(code, level)
            if not code:
                return "Auto-detect on: write or send a voice note in any language."
            lvl = {1: " · beginner", 2: " · intermediate", 3: " · advanced"}.get(level, "")
            name = _LANG_NAMES[code]
            return (f"Immersion: {name}{lvl}. I'll reply only in {name}. "
                    "Send /all to switch back.")
        return None

    async def _respond(self, update, user: _UserState, user_text: str) -> None:
        """Generate a reply (off the event loop), send text + a spoken voice note."""
        await update.message.chat.send_action("typing")
        reply = await asyncio.to_thread(lambda: "".join(user.session.stream(user_text)))
        reply = reply.strip()
        if not reply:
            return
        await update.message.reply_text(reply)
        audio = await asyncio.to_thread(render_reply, self.tts, reply)
        if audio is not None:
            ogg = await asyncio.to_thread(_to_ogg, audio[0], audio[1])
            try:
                with open(ogg, "rb") as f:
                    await update.message.reply_voice(f)
            finally:
                os.unlink(ogg)

    # --- handlers ---
    def _authorized(self, update) -> bool:
        allow = _allowed_ids()
        return not allow or (update.effective_user and update.effective_user.id in allow)

    async def on_command(self, update, context) -> None:
        if not self._authorized(update):
            return await update.message.reply_text("Not authorized.")
        word = update.message.text[1:].split()[0].split("@")[0].lower()
        if word in ("start", "help"):
            return await update.message.reply_text(
                "iBeto on Telegram. Send text or a voice note in any language.\n\n"
                + _help_text())
        msg = self._switch_language(self._user(update.effective_user.id), word)
        await update.message.reply_text(msg or "Unknown command. Send /help.")

    async def on_text(self, update, context) -> None:
        if not self._authorized(update):
            return await update.message.reply_text("Not authorized.")
        await self._respond(update, self._user(update.effective_user.id), update.message.text)

    async def on_voice(self, update, context) -> None:
        if not self._authorized(update):
            return await update.message.reply_text("Not authorized.")
        user = self._user(update.effective_user.id)
        tg_file = await update.message.voice.get_file()
        fd, path = tempfile.mkstemp(suffix=".oga")
        os.close(fd)
        try:
            await tg_file.download_to_drive(path)
            text = await asyncio.to_thread(self.stt.transcribe, path, user.stt_lang)
        finally:
            os.unlink(path)
        if not text.strip():
            return await update.message.reply_text("(didn't catch that)")
        await update.message.reply_text(f"You said: {text}")
        await self._respond(update, user, text)


def run() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Set TELEGRAM_BOT_TOKEN (from @BotFather). See ibeto/channels/telegram.py.")
        return 1
    from telegram.ext import Application, MessageHandler, filters

    channel = TelegramChannel()
    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.COMMAND, channel.on_command))
    app.add_handler(MessageHandler(filters.VOICE, channel.on_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, channel.on_text))
    allow = _allowed_ids()
    print(f"iBeto Telegram bot running (allowlist: {allow or 'ANYONE — set IBETO_TG_ALLOW'}).")
    print("Message your bot on Telegram. Ctrl-C to stop.")
    app.run_polling()
    return 0
