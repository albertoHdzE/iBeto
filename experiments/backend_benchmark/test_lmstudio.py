"""Manual check against a live LM Studio server (not collected by pytest)."""

from ibeto.llm.lmstudio import LMStudioBackend

backend = LMStudioBackend()

print("=" * 60)
print("Available models")
print("=" * 60)
for model in backend.list_models().data:
    print(model.id)

print()
print("=" * 60)
print("Streaming conversation")
print("=" * 60)

messages = [
    {"role": "user", "content": "Introduce yourself in German in two sentences."}
]
for delta in backend.stream(messages):
    print(delta, end="", flush=True)
print()
