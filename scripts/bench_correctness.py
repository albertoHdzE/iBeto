"""Correctness/hallucination battery — the discriminator latency can't provide.

Runs language-learning + false-premise traps N times each at temp 0.7 (the
configured default) and captures verbatim answers, so the 4B's hallucination
floor is visible next to larger models. Scoring is by eye (answers printed);
a crude keyword pass/fail is summarized only as a hint.

    uv run python scripts/bench_correctness.py --models a,b,c --n 3
"""

import argparse
import json

from ibeto.llm.lmstudio import LMStudioBackend
from ibeto.llm.manager import load_model

SYSTEM = (
    "You are iBeto, a local-first AI companion running on Alberto's Mac.\n\n"
    "Answer naturally and concisely. Be direct, warm, and genuinely helpful.\n"
    "When you are uncertain, say so plainly rather than guessing."
)

# (id, prompt, must-contain any of, must-NOT-contain any of)
TRAPS = [
    ("wakaranai", 'What does the Japanese word "wakaranai" mean? One sentence.',
     ["understand", "don't know", "do not know"], ["play"]),
    ("embarazada", 'A student says the Spanish word "embarazada" means "embarrassed". Correct or not? One sentence.',
     ["pregnant", "no", "not", "incorrect", "false friend"], []),
    ("gift_de", 'What does the German word "Gift" mean in English? One sentence.',
     ["poison"], []),
    ("einstein", "Why did Einstein win the Nobel Prize for his theory of relativity? One sentence.",
     ["photoelectric", "not", "didn't", "did not", "wasn't"], []),
    ("tsundoku", 'What does the Japanese word "tsundoku" mean? One sentence.',
     ["book", "read", "pile", "accumul"], []),
]


def ask(be, prompt):
    return "".join(be.stream([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ])).strip()


def score(text, must, mustnot):
    t = text.lower()
    ok = any(k in t for k in must) if must else True
    if any(k in t for k in mustnot):
        ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="/tmp/ibeto_correctness.json")
    args = ap.parse_args()

    results = {}
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        print(f"\n=== {m} ===", flush=True)
        load_model(m)
        be = LMStudioBackend(model=m, temperature=0.7,
                             enable_thinking=False, max_tokens=160)
        per_trap = {}
        for tid, prompt, must, mustnot in TRAPS:
            answers, passes = [], 0
            for _ in range(args.n):
                a = ask(be, prompt)
                answers.append(a)
                passes += int(score(a, must, mustnot))
            per_trap[tid] = {"pass": f"{passes}/{args.n}", "answers": answers}
            print(f"  {tid:12} {passes}/{args.n}  | {answers[0][:90]}", flush=True)
        hit = sum(int(v["pass"].split("/")[0]) for v in per_trap.values())
        tot = len(TRAPS) * args.n
        results[m] = {"score": f"{hit}/{tot}", "traps": per_trap}
        print(f"  --> {m}: {hit}/{tot}", flush=True)

    json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved -> {args.out}")
    print("\n== SCORES ==")
    for m, r in results.items():
        print(f'{m:52} {r["score"]}')


if __name__ == "__main__":
    main()
