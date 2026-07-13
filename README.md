# iBeto

A local-first conversational AI companion for Apple Silicon. Runs entirely on
your Mac against a local model served by LM Studio — no cloud, no API keys.

## Status

**Phase 1 + voice** — a usable streaming companion you can type or talk to:

- ✅ LM Studio backend (`chat` + `stream`)
- ✅ Interactive terminal chat with conversation history
- ✅ Persistent history — resume past conversations (`--resume`)
- ✅ Voice mode — push-to-talk speech in, spoken reply out (`--voice`)
- ✅ Vision — `/look` captures a camera frame and asks the model about it
- ✅ Model switching — `/model` lists/loads models via the `lms` CLI (auto-loads at startup)
- ✅ Tunable reasoning — `/think on|off` (off is ~4× faster for casual chat)
- ✅ System prompt from a Markdown file
- ✅ Configurable model / URL / temperature via `configs/ibeto.toml`
- ✅ Optional latency + tokens/sec metrics (`--stats`)

Runs a single vision-language model that handles both text and images. On the
Mac Studio (96 GB) larger models are the norm and multiple resident models are
possible; see `docs/HANDOFF.md`.

Deliberately postponed (added later, one capability at a time):
tools/automation, semantic long-term memory.

> **Moving machines?** `docs/HANDOFF.md` is the full transfer document (state,
> measurements, constraints, roadmap, acceptance tests) for continuing this
> project on different hardware.

## Requirements

- macOS + [uv](https://docs.astral.sh/uv/), Python 3.12
- [LM Studio](https://lmstudio.ai/) running with its local server started
  (Developer tab) and a model loaded

## Quickstart

```bash
./scripts/setup.sh     # sync deps, verify LM Studio is reachable
./scripts/ibeto        # launch the chat  (add --stats for metrics)
```

Or directly: `uv run ibeto` / `uv run python -m ibeto`.

## Handbook

Two ways to talk to iBeto, and they take commands differently:

- **Text mode** (`ibeto`) — you **type** commands like `/think` and `/look`.
- **Voice mode** (`ibeto --voice`) — your input is your voice, so you **speak**
  the commands out loud. Typed `/think` does nothing here.

### Launch flags

| Flag | Effect |
|------|--------|
| _(none)_ | Text chat |
| `--voice` | Push-to-talk voice: speech in, spoken reply out |
| `--lang CODE` / `-l` | Lock the spoken language for a focused practice session (e.g. `-l de`, `-l french`, `-l ja`); `all` = auto-detect the mix (default). Reduces short-phrase misdetection. |
| `--resume` | Continue the previous conversation |
| `--think` | Start with reasoning mode on (default off) |
| `--stats` | Show latency (TTFT) and tokens/sec after each reply |

Flags combine, e.g. `ibeto --voice --resume`.

### In-conversation commands

| Goal | Text mode (type) | Voice mode (say) |
|------|------------------|------------------|
| Reasoning on | `/think on` (or `/think` to toggle) | "think harder" |
| Reasoning off | `/think off` | "stop thinking" |
| Use the camera | `/look` or `/look <question>` | "look at this, what is it?" |
| List / switch model | `/model` or `/model gemma` | (type it) |
| Quit | `exit` or Ctrl-D | Ctrl-C |

In voice mode the gesture is: press **Enter** to start recording, speak, press
**Enter** again to stop. iBeto transcribes, replies on screen, and speaks aloud.
You can also **type** any command (or a whole message) at the `[Enter to speak]`
prompt instead of speaking — press Enter alone to record audio.

### Switching models

`/model` lists your downloaded models (current marked `*`); `/model <number>` or
`/model <name>` switches. On 8 GB this unloads the current model and loads the
new one (~15s) via the `lms` CLI. iBeto also auto-loads the configured model at
startup, so `ibeto` / `ibeto --voice` just works even if nothing is loaded yet.

### Voice mode



```bash
uv run ibeto --voice
```

Push-to-talk: press Enter to start speaking, Enter again to stop. iBeto
transcribes with Whisper, replies in the terminal, and speaks the reply aloud
with a **neural Kokoro voice** — sentence-by-sentence *as the reply streams*, so
it starts talking in ~0.8 s instead of waiting for the whole reply. Your iPhone
16 Pro works as the mic out of the box via Continuity. First run downloads the
Whisper model and the ~300 MB Kokoro model, and macOS asks for mic permission.

The voice engine is set by `tts_engine` in `configs/ibeto.toml`:

- **`xtts` (default)** — one natural neural voice (XTTS-v2) for **every** language
  (en/de/fr/es/it/pt/ja/zh/ar): the same speaker throughout, most consistent. It
  runs in an isolated `uv run --no-project` worker (`ibeto/audio/xtts_worker.py`)
  because coqui-tts pins `numpy<2` while the app needs `numpy>=2`. It is
  **self-configuring**: the first `ibeto --voice` builds that env and downloads
  the ~1.8 GB model once (a few minutes), and later launches load it in ~15 s —
  no setup step needed. Pick the speaker with `tts_xtts_speaker`.
- **`kokoro`** — fast native per-language voices (~0.15 RTF, no torch), a
  *different* voice per language. Snappier turn-taking; set this if XTTS feels slow.
- **`say`** — robotic macOS voice, zero deps (last resort).

**How a reply is spoken (both neural engines).** Each sentence is cleaned of
markdown/emoji, split into script runs, and Latin runs are language-detected (via
`lingua`, biased toward English so cognates aren't mispronounced). Each chunk is
resampled to one 24 kHz stream so switching languages mid-reply is seamless.
Under `kokoro` the routing uses per-language voices (`tts_voice`,
`tts_voice_es/fr/it/pt/zh/de/ar/ja`); under `xtts` the same voice speaks them all.

Limit: a foreign phrase *glued* into an English clause with no punctuation (e.g.
"you say Ich liebe dich") is voiced in the dominant language — telling apart
same-alphabet languages at the word level is a detection limit, not an engine
one. Full foreign sentences and comma/quote-set-off phrases route correctly.

**Speech input is multilingual too.** By default Whisper `large-v3-turbo`
**auto-detects** the spoken language (`stt_language = ""`), so you can just speak
English, French, Japanese, Spanish, etc. and be understood — no need to pre-set a
language. It's CPU-bound (no Metal), so it uses many cores (`whisper_threads`):
~2.5 s per utterance on the M3 Ultra. For faster-but-English-only turn-taking,
set `whisper_model = "base"` and `stt_language = "en"`.

### Vision

Type `/look` in a chat to capture a camera frame and ask about it, or
`/look <question>` for a specific question:

```
You > /look what is on my desk?
Looking...
Assistant > I see a keyboard, a coffee cup, and a notebook...
```

Requires a vision-language model loaded in LM Studio (the default
`qwen3.5-4b-instruct-revised` is one). The iPhone works as the camera via
Continuity; set `camera_index` in `configs/ibeto.toml` if the wrong camera is
picked. The image is sent for that turn only, not kept in history.

In voice mode, say a phrase like *"look at this, what is it?"* to trigger the
camera for that turn.

### Reasoning (thinking) mode

`qwen3.5-4b` reasons silently before answering, which is great for hard
questions but slow for chit-chat (~4× the latency). It defaults to **off**.

- Text: `/think on`, `/think off`, or `/think` to toggle.
- Voice: say *"think harder"* to enable, *"stop thinking"* to disable.
- Start with it on: `ibeto --think`. Default lives in `configs/ibeto.toml`.

Note: `qwen3.5-4b`'s thinking-off flag is only partly honored — it sometimes
still reasons at length. `max_tokens` in `configs/ibeto.toml` (default 800)
hard-caps every reply so nothing ever hangs for minutes. For reliably fast
replies, prefer a non-reasoning model (see Configuration).

```
iBeto v0.1
Connected to LM Studio  ·  model: qwen3.5-4b-instruct-revised
You > Hello
Assistant > Hello Alberto! How can I help today?
```

Type `exit` or press Ctrl-D to quit.

## Configuration

Everything tunable lives in `configs/ibeto.toml` — change the model there, not
in code. Default is `gemma-4-12b-it-mlx`, a vision-language model chosen by
measurement (`scripts/bench_*.py`): 15/15 on a hallucination/false-premise
battery, natural in ES/DE/JA/FR, ~0.53s TTFT and ~99 tok/s with a 256k context.
The chosen model must be vision-capable for `/look` to work on a single model.
System prompts live in `ibeto/prompts/*.md`.

**Switching for harder tasks.** `/model mistral-small-3.2-24b` swaps to a 24B
VLM live (~6s); the text-only reasoning models (the 27–35B qwen distills,
llama-3.3-70b) are available too but can't serve `/look`. On the 96 GB Mac
Studio, model size is no longer the constraint it was on the 8 GB M2.

## Layout

```
ibeto/
  llm/lmstudio.py    LM Studio communication only
  core/session.py    conversation history + streaming (UI-independent)
  cli/               terminal + voice loops
  audio/             stt (Whisper), tts (Kokoro+Piper, multilingual streaming), mic
  vision/            camera capture -> base64 for the vision-language model
  memory/            conversation persistence (save/load history)
  config/            TOML config loader
  prompts/           system prompts (Markdown) + loader
configs/ibeto.toml   model / URL / temperature / voice settings
scripts/             setup.sh, ibeto (launch)
```
