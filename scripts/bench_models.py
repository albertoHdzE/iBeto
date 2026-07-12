"""Benchmark vision-capable models faithfully through the iBeto backend.

Loads each candidate via the same `lms` swap the app uses, then measures TTFT
and tok/s with the identical streaming call (temperature, enable_thinking flag,
max_tokens). Runs three probes per model:
  1. plain reply   -> latency (TTFT + tok/s)
  2. wakaranai trap -> correctness go/no-go (verbatim answer captured)
  3. vision probe   -> a controlled synthetic image, tests the /look pipeline

Numbers match `--stats` exactly (see ibeto/cli/_print_stats). Run:
    uv run python scripts/bench_models.py --models a,b,c
"""

import argparse
import base64
import json
import time

import cv2
import numpy as np

from ibeto.llm.lmstudio import LMStudioBackend
from ibeto.llm.manager import load_model

SYSTEM = (
    "You are iBeto, a local-first AI companion running on Alberto's Mac.\n\n"
    "Answer naturally and concisely. Be direct, warm, and genuinely helpful.\n"
    "When you are uncertain, say so plainly rather than guessing."
)

PLAIN = "Hello! Introduce yourself in one short sentence."
WAKARANAI = 'What does the Japanese word "wakaranai" mean? Answer in one sentence.'
VISION_Q = "Describe this image: what text and what shapes do you see?"


def make_test_image() -> str:
    """A controlled scene: white bg, red circle, blue rectangle, text 'MILK 2%'."""
    img = np.full((360, 480, 3), 255, np.uint8)
    cv2.circle(img, (120, 180), 70, (0, 0, 255), -1)          # red (BGR)
    cv2.rectangle(img, (260, 110), (420, 250), (255, 0, 0), -1)  # blue
    cv2.putText(img, "MILK 2%", (150, 320), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (0, 0, 0), 3, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def measure(backend, messages):
    """Return (ttft, tok_s, completion_tokens, text) mirroring _print_stats."""
    started = time.perf_counter()
    first = None
    chunks = []
    for delta in backend.stream(messages):
        if first is None:
            first = time.perf_counter()
        chunks.append(delta)
    now = time.perf_counter()
    usage = backend.last_usage
    ttft = (first - started) if first else None
    ctoks = usage.completion_tokens if usage else None
    tok_s = (ctoks / (now - first)) if (usage and first and now > first) else None
    return ttft, tok_s, ctoks, "".join(chunks)


def bench(model: str, image_url: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    load_model(model)          # unload --all + load, exactly like /model
    swap_s = time.perf_counter() - t0

    be = LMStudioBackend(model=model, temperature=0.7,
                         enable_thinking=False, max_tokens=max_tokens)

    def msg(user, img=None):
        content = user if img is None else [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": img}},
        ]
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": content}]

    ttft, tok_s, _, plain = measure(be, msg(PLAIN))
    _, _, _, waka = measure(be, msg(WAKARANAI))
    v_ttft, _, _, vis = measure(be, msg(VISION_Q, image_url))

    return {
        "model": model,
        "swap_s": round(swap_s, 1),
        "ttft_s": round(ttft, 2) if ttft else None,
        "tok_s": round(tok_s) if tok_s else None,
        "plain": plain.strip(),
        "wakaranai": waka.strip(),
        "vision_ttft_s": round(v_ttft, 2) if v_ttft else None,
        "vision": vis.strip(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--out", default="/tmp/ibeto_bench.json")
    args = ap.parse_args()

    image_url = make_test_image()
    results = []
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        print(f"\n=== {m} ===", flush=True)
        try:
            r = bench(m, image_url, args.max_tokens)
        except Exception as e:
            r = {"model": m, "error": str(e)}
        results.append(r)
        print(json.dumps(r, indent=2, ensure_ascii=False), flush=True)

    json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved -> {args.out}", flush=True)

    print("\n== SUMMARY ==")
    print(f'{"model":52} {"swap":>6} {"ttft":>6} {"tok/s":>6}')
    for r in results:
        if "error" in r:
            print(f'{r["model"]:52} ERROR {r["error"][:40]}')
            continue
        na = lambda v: "n/a" if v is None else v
        print(f'{r["model"]:52} {na(r["swap_s"]):>6} {na(r["ttft_s"]):>6} {na(r["tok_s"]):>6}')


if __name__ == "__main__":
    main()
