"""
Canonical face alignment — a direct port of `Face ID/FaceAligner.swift`.

ArcFace needs faces warped into a fixed pose (eyes level, at fixed positions);
a loose crop measurably degrades it. Same 112x112 template, same tiering:
five points, else eyes-only, else a padded crop.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

OUTPUT_SIZE = 112

# Standard ArcFace 112x112 template: left eye, right eye, nose, left mouth,
# right mouth. "Left"/"right" are on-screen, not anatomical. Byte-for-byte the
# same constants as `FaceAligner.referencePoints` in the Swift build.
REFERENCE_POINTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
EYE_REFERENCE_POINTS = REFERENCE_POINTS[:2]

FIVE_POINT = "5-point"
TWO_POINT = "2-point (eyes only)"
PADDED_CROP = "padded crop (no alignment)"


@dataclass
class AlignedFace:
    image: np.ndarray  # 112x112 BGR
    tier: str


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Least-squares similarity (rotate + uniform scale + translate), 2x3.

    `cv2.estimateAffinePartial2D` solves exactly this. RANSAC is pointless on
    five hand-picked correspondences, so use LMEDS over all of them.
    """
    if src.shape[0] < 2:
        return None
    matrix, _ = cv2.estimateAffinePartial2D(
        src.astype(np.float32), dst.astype(np.float32), method=cv2.LMEDS
    )
    return matrix


def align(image_bgr: np.ndarray, landmarks: np.ndarray, box: np.ndarray | None = None) -> AlignedFace | None:
    """Best-effort alignment, mirroring the Swift tiering."""
    if landmarks is not None and landmarks.shape[0] >= 5:
        matrix = _similarity_transform(landmarks[:5], REFERENCE_POINTS)
        if matrix is not None:
            warped = cv2.warpAffine(
                image_bgr, matrix, (OUTPUT_SIZE, OUTPUT_SIZE), flags=cv2.INTER_LINEAR
            )
            return AlignedFace(warped, FIVE_POINT)

    if landmarks is not None and landmarks.shape[0] >= 2:
        matrix = _similarity_transform(landmarks[:2], EYE_REFERENCE_POINTS)
        if matrix is not None:
            warped = cv2.warpAffine(
                image_bgr, matrix, (OUTPUT_SIZE, OUTPUT_SIZE), flags=cv2.INTER_LINEAR
            )
            return AlignedFace(warped, TWO_POINT)

    if box is None:
        return None
    cropped = crop(image_bgr, box)
    if cropped is None or cropped.size == 0:
        return None
    resized = cv2.resize(cropped, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return AlignedFace(resized, PADDED_CROP)


def crop(image_bgr: np.ndarray, box: np.ndarray, padding_fraction: float = 0.2) -> np.ndarray | None:
    """Port of `FaceDetector.crop` — pads out so the embedder sees some context."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding_fraction
    pad_y = (y2 - y1) * padding_fraction
    x1 = int(max(0, x1 - pad_x))
    y1 = int(max(0, y1 - pad_y))
    x2 = int(min(w, x2 + pad_x))
    y2 = int(min(h, y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return image_bgr[y1:y2, x1:x2]
