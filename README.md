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
- ✅ Tunable reasoning — `/think on|off` (off is ~4× faster for casual chat)
- ✅ System prompt from a Markdown file
- ✅ Configurable model / URL / temperature via `configs/ibeto.toml`
- ✅ Optional latency + tokens/sec metrics (`--stats`)

Runs a single model at a time (tuned for 8 GB RAM): the same loaded
vision-language model handles both text and images.

Deliberately postponed (added later, one capability at a time):
tools/automation, semantic long-term memory.

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
| Quit | `exit` or Ctrl-D | Ctrl-C |

In voice mode the gesture is: press **Enter** to start recording, speak, press
**Enter** again to stop. iBeto transcribes, replies on screen, and speaks aloud.

### Voice mode



```bash
uv run ibeto --voice
```

Push-to-talk: press Enter to start speaking, Enter again to stop. iBeto
transcribes with Whisper, replies in the terminal, and speaks the reply aloud.
Your iPhone 16 Pro works as the mic out of the box via Continuity. First run
downloads the Whisper model and macOS asks for microphone permission.

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

```
iBeto v0.1
Connected to LM Studio  ·  model: qwen3.5-4b-instruct-revised
You > Hello
Assistant > Hello Alberto! How can I help today?
```

Type `exit` or press Ctrl-D to quit.

## Configuration

Everything tunable lives in `configs/ibeto.toml` — change the model there, not
in code. Default is `qwen3.5-4b-instruct-revised` (proven safe on an 8 GB M2;
larger models have frozen it). Only one model loads at a time on 8 GB, so the
chosen model must be a vision-language model for `/look` to work. System prompts
live in `ibeto/prompts/*.md`.

## Layout

```
ibeto/
  llm/lmstudio.py    LM Studio communication only
  core/session.py    conversation history + streaming (UI-independent)
  cli/               terminal + voice loops
  audio/             stt (Whisper), tts (macOS say), mic (push-to-talk)
  vision/            camera capture -> base64 for the vision-language model
  memory/            conversation persistence (save/load history)
  config/            TOML config loader
  prompts/           system prompts (Markdown) + loader
configs/ibeto.toml   model / URL / temperature / voice settings
scripts/             setup.sh, ibeto (launch)
```
