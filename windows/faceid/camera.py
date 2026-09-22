"""Camera capture — the Windows counterpart to `CameraManager.swift`."""

from __future__ import annotations

import sys

import cv2


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        # CAP_DSHOW avoids the multi-second MSMF startup stall on many Windows
        # webcams; elsewhere let OpenCV pick its own backend.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {index}. On Windows check "
                "Settings > Privacy & security > Camera, and close any app already using it."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def list_cameras(limit: int = 8) -> list:
    """Indices that open successfully. OpenCV has no device enumeration API,
    so this probes — noisy on some drivers, which is why it is opt-in.
    """
    found = []
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    for i in range(limit):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found
