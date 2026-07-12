"""Measure time-to-first-audio: streaming Kokoro vs the old speak-after-reply.

Drives a real model reply through SentenceSpeaker and records when the FIRST
audio starts vs when the FULL reply finishes generating (= when macOS `say`
used to start). Plays the reply aloud through the real pipeline.
"""
import time

from ibeto.audio.tts import SentenceSpeaker, make_tts
from ibeto.config import load_config
from ibeto.core.session import ConversationSession
from ibeto.llm.lmstudio import LMStudioBackend
from ibeto.llm.manager import load_model
from ibeto.prompts import load_prompt

cfg = load_config()
load_model(cfg.model)  # ensure the configured model is the one loaded

tts = make_tts(cfg)
marks = {}
_orig = tts.speak
def timed(text):
    marks.setdefault("first_audio", time.perf_counter())
    _orig(text)
tts.speak = timed

spk = SentenceSpeaker(tts)
backend = LMStudioBackend(model=cfg.model, temperature=0.7,
                          enable_thinking=False, max_tokens=cfg.max_tokens)
session = ConversationSession(backend, load_prompt(cfg.system_prompt))

start = time.perf_counter()
first_tok = None
out = []
prompt = ("Tell me about the history and appeal of running AI locally, "
          "in about 120 words, as a few flowing sentences.")
for d in session.stream(prompt):
    if first_tok is None:
        first_tok = time.perf_counter()
    out.append(d)
    spk.feed(d)
gen_done = time.perf_counter()   # old `say` would START speaking here
spk.finish()
all_done = time.perf_counter()
spk.close()

ttft = first_tok - start
first_audio = marks["first_audio"] - start
print("\n--- TTS latency ---")
print(f"reply: {''.join(out).strip()}")
print(f"TTFT (first token):            {ttft:5.2f}s")
print(f"NEW first audio (streaming):   {first_audio:5.2f}s")
print(f"OLD first audio (say @ gen end): {gen_done - start:5.2f}s")
print(f"perceived latency cut:         {gen_done - marks['first_audio']:5.2f}s")
print(f"full reply spoken by:          {all_done - start:5.2f}s")
