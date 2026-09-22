"""
Liveness / anti-spoofing.

The macOS build runs five cues (`Face ID/Liveness/`): gloss-glare and
device-detected act as DENY signals, while flat-vs-3D, depth-pose and blink act
as CONFIRM signals. This module reimplements the ones that survive the move to
SCRFD.

WHAT IS HERE, AND WHAT IS NOT — read this before trusting it:

  gloss / glare     PORTED IN SPIRIT. Pure pixel work, so it transfers. A large
                    flat specular highlight is glass or a screen, not skin.
  device detected   PORTED IN SPIRIT. Looks for a strong rectangular edge
                    enclosing the face — a phone or tablet held up.
  depth / pose      PORTED. Needs only the five points SCRFD gives: if the nose
                    offset tracks head yaw across frames, the face has real depth.
  motion            NEW, replacing part of flat-vs-3D. A photo held by hand
                    translates rigidly; a real head changes its landmark
                    geometry. Cheap, and catches static prints.
  blink             *** MISSING ***. Eye-aspect-ratio needs per-eyelid contours.
                    SCRFD emits one point per eye, so EAR cannot be computed.
                    Restoring it requires InsightFace's 2d106det landmark model.
  flat vs 3D        *** PARTIAL ***. The macOS plane-fit uses dense landmarks;
                    with five points only the depth/pose proxy above remains.

Because two of the five cues are absent or weakened, THIS IS MEASURABLY EASIER
TO SPOOF THAN THE macOS BUILD. It is not a finished anti-spoofing system.
Thresholds here are first-pass values chosen by inspection, not tuned against a
spoof corpus.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

LIGHT = "light"
HEAVY = "heavy"


@dataclass
class LivenessSnapshot:
    passed: bool
    score: float
    reasons: list           # human-readable, for the CLI and logs
    denied_by: list         # cues that actively rejected
    confirmed_by: list      # cues that actively vouched


class LivenessAnalyzer:
    """Rolling-window driver, mirroring `LivenessAnalyzer.swift`'s shape:
    observe() per frame, decide from the accumulated window.
    """

    def __init__(self, mode: str = LIGHT, window: int = 24) -> None:
        self.mode = mode
        self.window = window
        self._yaws: deque = deque(maxlen=window)
        self._nose_offsets: deque = deque(maxlen=window)
        self._landmark_sets: deque = deque(maxlen=window)
        self._glare_hits = 0
        self._device_hits = 0
        self._frames = 0

    def reset(self) -> None:
        self._yaws.clear()
        self._nose_offsets.clear()
        self._landmark_sets.clear()
        self._glare_hits = self._device_hits = self._frames = 0

    def observe(self, frame_bgr: np.ndarray, face) -> None:
        self._frames += 1
        yaw, _, _ = face.pose()
        le, re, nose = face.landmarks[0], face.landmarks[1], face.landmarks[2]
        eye_mid = (le + re) / 2.0
        eye_dist = float(np.linalg.norm(re - le)) or 1.0
        # Signed, scale-free nose displacement along the eye axis.
        offset = float(np.dot(nose - eye_mid, (re - le) / eye_dist)) / eye_dist
        self._yaws.append(yaw)
        self._nose_offsets.append(offset)
        self._landmark_sets.append(face.landmarks.copy() / eye_dist)

        if _glare(frame_bgr, face):
            self._glare_hits += 1
        if _device_rectangle(frame_bgr, face):
            self._device_hits += 1

    # --- cues --------------------------------------------------------------

    def _depth_pose(self) -> tuple[bool, float]:
        """Nose offset should track yaw. On a flat photo the nose is painted on,
        so turning the print moves everything together and the correlation
        collapses.
        """
        if len(self._yaws) < 8:
            return False, 0.0
        y = np.asarray(self._yaws, dtype=np.float64)
        o = np.asarray(self._nose_offsets, dtype=np.float64)
        if y.std() < math.radians(4.0):
            return False, 0.0  # not enough head movement to judge
        corr = float(np.corrcoef(y, o)[0, 1])
        if not math.isfinite(corr):
            return False, 0.0
        return corr > 0.55, corr

    def _motion(self) -> tuple[bool, float]:
        """Non-rigid change in normalized landmark geometry."""
        if len(self._landmark_sets) < 8:
            return False, 0.0
        stack = np.stack(self._landmark_sets)
        centered = stack - stack.mean(axis=1, keepdims=True)
        variation = float(centered.std(axis=0).mean())
        return variation > 0.012, variation

    def snapshot(self) -> LivenessSnapshot:
        reasons, denied, confirmed = [], [], []

        glare_rate = self._glare_hits / max(1, self._frames)
        device_rate = self._device_hits / max(1, self._frames)
        if glare_rate > 0.35:
            denied.append("glossGlare")
            reasons.append(f"large flat specular highlight on {glare_rate:.0%} of frames — glass or a screen")
        if device_rate > 0.30:
            denied.append("deviceDetected")
            reasons.append(f"device-shaped rectangle around the face on {device_rate:.0%} of frames")

        depth_ok, corr = self._depth_pose()
        if depth_ok:
            confirmed.append("depthPose")
            reasons.append(f"nose offset tracks yaw (r={corr:.2f}) — the face has real depth")
        motion_ok, variation = self._motion()
        if motion_ok:
            confirmed.append("motion")
            reasons.append(f"non-rigid landmark motion ({variation:.4f})")

        if denied:
            return LivenessSnapshot(False, 0.0, reasons, denied, confirmed)

        if self.mode == HEAVY:
            # Heavy demands positive proof of life, not just absence of tells.
            passed = bool(confirmed)
            if not passed:
                reasons.append("heavy mode: no positive liveness cue fired — move your head slightly")
        else:
            passed = True

        score = min(1.0, 0.5 * len(confirmed) + (0.5 if not denied else 0.0))
        return LivenessSnapshot(passed, score, reasons, denied, confirmed)


# --- pixel cues ------------------------------------------------------------

def _face_roi(frame_bgr: np.ndarray, face) -> np.ndarray | None:
    x1, y1, x2, y2 = [int(v) for v in face.box]
    roi = frame_bgr[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
    return roi if roi.size else None


def _glare(frame_bgr: np.ndarray, face) -> bool:
    """A screen or a print behind glass throws one big, flat, blown-out patch.
    Skin throws many small scattered highlights instead.
    """
    roi = _face_roi(frame_bgr, face)
    if roi is None:
        return False
    v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]
    bright = (v > 240).astype(np.uint8)
    if bright.sum() < 40:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return False
    largest = max(stats[1:, cv2.CC_STAT_AREA])
    return bool(largest / float(roi.shape[0] * roi.shape[1]) > 0.035)


def _device_rectangle(frame_bgr: np.ndarray, face) -> bool:
    """A phone or tablet held up puts a strong quadrilateral around the face."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fx1, fy1, fx2, fy2 = face.box
    for contour in contours:
        if cv2.contourArea(contour) < face.area * 1.15:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue
        x, y, cw, ch = cv2.boundingRect(approx)
        if x <= fx1 and y <= fy1 and x + cw >= fx2 and y + ch >= fy2:
            if cw * ch < 0.92 * w * h:  # ignore the frame border itself
                return True
    return False
