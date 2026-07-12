# iBeto — Handoff: Mac mini M2 (8 GB) → Mac Studio M3 Ultra (96 GB)

Written on the M2 at tag `v0.1.0-pivot`. This is the complete transfer of
knowledge needed to clone the repo on the Mac Studio and continue without
re-deriving anything. Read it top to bottom before touching code.

---

## 1. What iBeto is

A local-first, multimodal AI companion. Everything runs on Alberto's Mac against
a model served by **LM Studio** — no cloud, no API keys. The long-term goal is an
embodied companion ("Reachy mini level"): natural voice, fast turn-taking, eyes,
memory, personality. The terminal app is the brain; the body comes later.

The original design conversation is in `docs/research/chatGPT-proposal.txt`.

## 2. How Alberto wants it built (non-negotiable)

- **No overengineering.** Add an abstraction only when a second use appears.
- **One capability per milestone.** The repo must run after every commit.
- **Measure, don't assume.** He reads LM Studio's logs and will check your
  claims. Never say something is fast/working without a number or a run.
- **Everything user-tunable at runtime**, not hardcoded (reasoning on/off,
  vision on demand, model switching).
- **Commits are authored by Alberto only.** Do **not** add a Claude co-author
  trailer. Commit and push after each milestone.
- Token-efficient communication. No filler.

## 3. State at handoff (tag `v0.1.0-pivot`)

Working end-to-end, all committed and pushed:

| Capability | How | File |
|---|---|---|
| Streaming chat | OpenAI SDK against LM Studio | `ibeto/llm/lmstudio.py` |
| Conversation core (UI-independent) | `ConversationSession.stream()` | `ibeto/core/session.py` |
| Text + voice loops, slash commands | `_handle_slash()` dispatcher | `ibeto/cli/__init__.py` |
| Push-to-talk voice | faster-whisper (CPU int8) + macOS `say` | `ibeto/audio/` |
| Vision on demand (`/look`) | OpenCV frame → base64 JPEG data URL | `ibeto/vision/capture.py` |
| Live model switching (`/model`) | `lms` CLI: unload --all, then load | `ibeto/llm/manager.py` |
| Tunable reasoning (`/think`) | mutable `backend.enable_thinking` | `ibeto/llm/lmstudio.py` |
| Persistent history (`--resume`) | JSON, user/assistant turns only | `ibeto/memory/` |
| Config surface | one commented TOML | `configs/ibeto.toml` |
| Tests | 10 smoke tests, no live server needed | `tests/test_smoke.py` |

Commands: `/think on|off`, `/look [question]`, `/model [name|number]`, `exit`.
Flags: `--voice`, `--think`, `--stats`, `--resume`.
Launch: `./scripts/setup.sh` then `./scripts/ibeto` (a zsh alias `ibeto` pointed
at `scripts/ibeto` existed on the M2 — recreate it on the Studio).

## 4. Measurements taken on the M2 (8 GB) — the evidence base

These numbers are why the code looks the way it does. Re-measure on the Studio;
most of the constraints they encode are expected to **disappear**.

- `qwen3.5-4b-instruct-revised`: reasons **silently** before answering. Thinking
  OFF → 15.5 s / 330 tokens. Thinking ON → 62.5 s / 1397 tokens. One runaway hit
  **143 s / 3182 tokens** *despite* `chat_template_kwargs: {enable_thinking:
  false}` — that flag is **not reliably honored** by this model. That is why
  `max_tokens` (default 800) exists as a seatbelt.
- `google/gemma-3-4b`: **0.6 s** reply, TTFT 0.52 s, non-reasoning, and still
  vision-capable (`"type": "vlm"` in LM Studio's native API). It became the
  default purely on that measurement.
- Generation rate on M2: ~22–23 tok/s. The latency was hidden reasoning, not
  weak hardware.
- STT: Whisper `base` with `language=None` misheard English as Japanese. Fixed by
  forcing `stt_language = "en"`. faster-whisper runs on **CPU int8** — there is
  no Metal backend, so it does not benefit from the GPU.
- `lms` model swap (unload all + load) takes ~14–15 s.

## 5. Known weaknesses at handoff — the reason for the hardware jump

1. **Not smart enough.** gemma-3-4b hallucinates. Live example: it told Alberto
   *"Wakaranai" means "Let's play"* (it means "I don't understand"). A 4B model
   is the ceiling on 8 GB, and it is too low for a trustworthy companion,
   especially for language learning.
2. **Robotic voice.** macOS `say` sounds like a 2005 GPS. Needs a neural TTS.
3. **Turn-taking is clunky.** Push-to-talk with Enter. No VAD, no barge-in.
4. **No long-term memory.** History is a flat JSON of the last conversation.

Problems 1 and 2 are *independent* — do not conflate them. A bigger model does
not fix the voice; a better voice does not fix hallucination.

## 6. What changes on the M3 Ultra (96 GB)

The single hardest constraint in the codebase — *only one small model fits* —
is gone. Concretely:

- **Raise the model.** With 96 GB you can hold a 27B–70B class model in memory
  and still have room. Candidates to benchmark (must be **vision-capable** if you
  want `/look` to keep working on a single model): `gemma-3-27b-it`,
  `mistral-small-3.x-24b` (VLM), `qwen3-vl` 30B+. Set it in `configs/ibeto.toml`
  and/or switch live with `/model` — that mechanism already works and needs no
  change.
- **Two models at once become possible.** `ibeto/llm/manager.py` currently calls
  `lms unload --all` before every load *because of the 8 GB limit*. On the Studio
  you may want a fast small model for chat plus a VLM for vision, or an LLM plus
  a TTS/embedding model resident simultaneously. **Do not rip this out
  speculatively** — only relax it when a milestone actually needs two models.
- **Whisper gets cheap.** Move `whisper_model` from `base` to `large-v3`;
  transcription accuracy was a real source of bugs. Still CPU-bound, but the
  Ultra has the cores.
- **`max_tokens = 800` was a seatbelt against slow hardware + runaway reasoning.**
  With a fast machine, raise it (e.g. 2048) once you confirm the chosen model
  respects the thinking flag. Keep *some* cap.
- **Reasoning becomes affordable.** `/think on` was painful at 22 tok/s. Re-measure;
  the default may flip.

**First thing to do on the Studio: re-run the benchmarks.** Every default in
`configs/ibeto.toml` was chosen from an 8 GB measurement and is now suspect.

## 7. Roadmap to "Reachy mini level"

In priority order. One per milestone, repo runnable after each.

1. **Neural streaming TTS (Kokoro).** Replace macOS `say` behind the existing
   `ibeto/audio/tts.py: speak(text, voice)` seam — it was deliberately isolated
   for exactly this. Also **speak sentence-by-sentence as the reply streams**
   instead of waiting for the full reply; that alone removes most of the felt
   latency. *This is the agreed next step and the biggest perceived-quality win.*
2. **Bigger model.** The hardware jump. Fixes "not smart enough". Benchmark
   candidates from §6 and set the new default from data.
3. **VAD + barge-in.** Kill push-to-talk: detect speech start/stop automatically
   (webrtcvad / silero-vad) and let Alberto interrupt the reply mid-sentence.
   This is what makes a conversation feel alive.
4. **Grounding / RAG** for language learning, so it stops inventing meanings.
5. **Long-term memory** — semantic recall across sessions, beyond flat history.
6. **Persona + proactivity** — it should have a character and sometimes speak first.
7. **Ambient vision** — periodic frames instead of on-demand `/look`.

**AG-UI**: Alberto asked about it. It is a protocol for rich GUI frontends; it is
premature while iBeto is a terminal app. `ConversationSession` is already
UI-independent, so adopting it later is additive, not a rewrite. Revisit when a
GUI exists.

## 8. Setup on the Mac Studio (do this first)

```bash
git clone https://github.com/albertoHdzE/iBeto.git
cd iBeto
./scripts/setup.sh            # uv sync + verify LM Studio is reachable
```

Then:

1. Install LM Studio, start its local server (Developer tab), and confirm the
   `lms` CLI exists at `~/.lmstudio/bin/lms` (`ibeto/llm/manager.py` hardcodes
   that path).
2. Download at least one vision-capable model (see §6).
3. Optional alias: `alias ibeto="$HOME/<path>/iBeto/scripts/ibeto"` in `~/.zshrc`.
4. Grant microphone + camera permissions on first voice/`/look` run.
5. iPhone 16 Pro is the mic/camera via Continuity (it appeared as "Espinosa phone
   Microphone"). `camera_index` in the TOML picks the camera if the wrong one is used.

## 9. Acceptance tests on the new machine (run these, report numbers)

Same tests as on the M2, so the two machines are comparable.

```bash
uv run pytest -q                       # 10 smoke tests, no server needed
uv run ibeto --stats                   # text: TTFT + tok/s per reply
uv run ibeto --voice --stats           # voice round-trip
```

In a chat, verify each of these and record the numbers:

- [ ] `pytest` — 10/10 pass.
- [ ] Plain reply — record **TTFT and tok/s**. (M2 baseline: gemma-3-4b, TTFT
      0.52 s, ~22 tok/s.)
- [ ] `/model` lists models; `/model <name>` swaps successfully. Record swap time.
      (M2 baseline: ~14 s.)
- [ ] `/think on` then a hard question — record tokens and wall time; confirm the
      chosen model actually honors thinking-off when toggled back.
- [ ] `/look what is on my desk?` — a correct description from the camera.
- [ ] Voice: Enter → speak → Enter, transcription correct, reply spoken.
- [ ] **The Japanese trap** — ask it what *"wakaranai"* means. gemma-3-4b said
      "Let's play". The correct answer is "I don't understand." A model that gets
      this wrong is not good enough; this is the go/no-go for the model upgrade.

## 10. Gotchas learned the hard way

- `uv run ibeto` only works **inside the project directory**. That's why
  `scripts/ibeto` `cd`s first — use the script or the alias, never bare `uv run`
  from `~`.
- `/model` is typed **inside the running app**, not passed as a shell argument
  (`ibeto /model` is an error).
- The image is stripped from stored history after a `/look` turn
  (`ConversationSession.stream`) to spare the context window — keep that if you
  raise the model, or revisit deliberately.
- Embedding models are filtered out of `/model`'s list (`"embed" not in id`).
- Reasoning flags are model-specific and sometimes ignored. Always keep a
  `max_tokens` cap.
