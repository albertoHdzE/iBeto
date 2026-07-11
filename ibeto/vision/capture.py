"""Grab a single frame from the camera as a base64 JPEG data URL.

On-demand only: iBeto looks when asked, so one model serves both text and
vision on limited RAM. The iPhone works as the camera via Continuity Camera.
"""

import base64

import cv2


def frame_to_data_url(bgr_frame, quality: int = 85) -> str:
    """Encode a BGR OpenCV frame as a data:image/jpeg;base64 URL."""
    ok, buffer = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode the camera frame.")
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def capture_frame(camera_index: int = 0) -> str:
    """Open the camera, grab one frame, and return it as a data URL."""
    camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open camera index {camera_index}. "
                "Set 'camera_index' in configs/ibeto.toml (try 0 or 1)."
            )
        # Discard a few frames so exposure/focus settle before capture.
        frame = None
        for _ in range(5):
            ok, frame = camera.read()
        if frame is None or not ok:
            raise RuntimeError("Camera opened but returned no frame.")
        return frame_to_data_url(frame)
    finally:
        camera.release()
