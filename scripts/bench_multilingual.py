"""Multilingual conversation probe — the assistant's #1 goal is natural,
multi-language chat, which the correctness battery (English prompts) didn't test.

For each language: (a) respond naturally to a casual message IN that language,
(b) gently correct a learner's mistake. Answers are captured verbatim for
eyeball judgement of fluency/naturalness; a crude in-language check is a hint
only. Compares the two top correctness models.

    uv run python scripts/bench_multilingual.py --models a,b
"""

import argparse
import json

from ibeto.llm.lmstudio import LMStudioBackend
from ibeto.llm.manager import load_model

SYSTEM = (
    "You are iBeto, a local-first AI companion and language tutor running on "
    "Alberto's Mac. Reply in the same language the user writes in. Be natural, "
    "warm, and concise. If the user makes a language mistake, gently correct it."
)

# (id, casual message, learner mistake to correct)
PROBES = [
    ("es", "¡Hola! ¿Qué tal tu día? Cuéntame algo interesante.",
     "Corrige con cariño: «Yo soy 25 años y ayer yo iba al cine con mis amigo.»"),
    ("de", "Hallo! Wie geht's dir heute? Erzähl mir etwas Schönes.",
     "Korrigiere freundlich: «Ich habe gestern in die Schule gegangen und habe viel Spaß gehabt.»"),
    ("ja", "こんにちは！今日はどんな一日でしたか？何か面白い話をしてください。",
     "優しく直してください：「昨日、私は友達と映画を見に行きました、とても楽しいでした。」"),
    ("fr", "Salut ! Comment s'est passée ta journée ? Raconte-moi quelque chose.",
     "Corrige gentiment : « Hier je suis allé au marché et j'ai acheté des pain et des lait. »"),
]


def ask(be, prompt):
    return "".join(be.stream([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ])).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", default="/tmp/ibeto_multilingual.json")
    args = ap.parse_args()

    results = {}
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        print(f"\n=== {m} ===", flush=True)
        load_model(m)
        be = LMStudioBackend(model=m, temperature=0.7,
                             enable_thinking=False, max_tokens=220)
        lang = {}
        for lid, casual, correct in PROBES:
            chat = ask(be, casual)
            corr = ask(be, correct)
            lang[lid] = {"chat": chat, "correction": corr}
            print(f"\n  [{lid}] casual:\n    {chat}", flush=True)
            print(f"  [{lid}] correction:\n    {corr}", flush=True)
        results[m] = lang

    json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
