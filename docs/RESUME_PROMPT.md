# Resume prompt (paste this into Claude Code on the Mac Studio)

Run `claude` inside the cloned `iBeto` repo and paste the block below as the
first message.

---

You are taking over **iBeto**, a local-first multimodal AI companion I built on a
Mac mini M2 (8 GB). We have just migrated to this **Mac Studio M3 Ultra, 96 GB
RAM**. The repo is at tag `v0.1.0-pivot` and runs end-to-end.

**Read `docs/HANDOFF.md` first** — it is a complete transfer document written by
the previous instance: project state, the measurements behind every default, the
known weaknesses, what changes now that the 8 GB one-model constraint is gone,
the roadmap, and the acceptance tests. Then read `README.md` and
`configs/ibeto.toml`.

How I want you to work (this matters more than speed):
- No overengineering. One capability per milestone; the repo must run after each.
- Measure, don't assume — I read the LM Studio logs and I will check your numbers.
- Everything tunable at runtime, not hardcoded.
- Commit and push after each milestone, **authored by me only — never add a Claude
  co-author trailer**.
- Be token-efficient. No filler.

Your first task, before writing any feature code:

1. Get the environment running here — `./scripts/setup.sh`, LM Studio server up,
   `lms` CLI present at `~/.lmstudio/bin/lms`.
2. Run the acceptance tests in `docs/HANDOFF.md` §9 and **report the numbers**
   next to the M2 baselines. Include the "wakaranai" trap — the old 4B model
   failed it.
3. Every default in `configs/ibeto.toml` was chosen under the 8 GB limit and is
   now suspect. Benchmark vision-capable models in the 24B–70B class and
   recommend a new default **from data**, not from reputation.

Then we do the roadmap in order, starting with **neural streaming TTS (Kokoro)** to
replace the robotic macOS `say`, speaking sentence-by-sentence as the reply
streams. Tell me what you find before you change anything.
