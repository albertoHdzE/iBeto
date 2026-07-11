"""Probe: send a synthetic image with known content through the real session
to the loaded VLM, verifying the end-to-end vision path (no camera needed)."""

import cv2
import numpy as np

from ibeto.config import load_config
from ibeto.core.session import ConversationSession
from ibeto.llm.lmstudio import LMStudioBackend
from ibeto.vision.capture import frame_to_data_url

# White canvas with a red circle and the word "CAT".
img = np.full((240, 320, 3), 255, dtype=np.uint8)
cv2.circle(img, (90, 120), 50, (0, 0, 255), -1)  # BGR red
cv2.putText(img, "CAT", (150, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

image = frame_to_data_url(img)

cfg = load_config()
backend = LMStudioBackend(base_url=cfg.base_url, model=cfg.model, temperature=cfg.temperature)
session = ConversationSession(backend, "You are a precise image describer.")

print("Model reply:")
for delta in session.stream("What shapes, colors, and text are in this image?", image=image):
    print(delta, end="", flush=True)
print()

# Verify the image was stripped from stored history afterward.
user_msg = session.messages[-2]
print("\nStored user message content:", repr(user_msg["content"]))
