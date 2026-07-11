"""Measure hidden-reasoning cost: completion_tokens + wall time for a short
prompt under different ways of disabling qwen3.5 thinking mode."""

import time

from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
MODEL = "qwen3.5-4b-instruct-revised"
Q = "Hello, how are you?"


def measure(label, messages, extra_body=None):
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.7,
        extra_body=extra_body or {},
    )
    dt = time.perf_counter() - t0
    usage = resp.usage
    reply = resp.choices[0].message.content or ""
    print(f"\n### {label}")
    print(f"  time={dt:.1f}s  completion_tokens={usage.completion_tokens}")
    print(f"  reply[:120]={reply[:120]!r}")


base_sys = "You are iBeto. Answer concisely."

measure("A. baseline", [
    {"role": "system", "content": base_sys},
    {"role": "user", "content": Q},
])

measure("B. /no_think in system prompt", [
    {"role": "system", "content": base_sys + " /no_think"},
    {"role": "user", "content": Q},
])

measure("C. chat_template_kwargs enable_thinking=false", [
    {"role": "system", "content": base_sys},
    {"role": "user", "content": Q},
], extra_body={"chat_template_kwargs": {"enable_thinking": False}})
