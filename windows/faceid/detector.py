"""
SCRFD face detection.

Replaces Vision's `VNDetectFaceRectanglesRequest` + `VNDetectFaceLandmarksRequest`
from the macOS build. SCRFD emits a box, a confidence and five landmarks
(both eyes, nose tip, both mouth corners) in one pass, which is exactly the
five points `FaceAligner` needs.

Two things Vision gave the macOS build for free are NOT available here and are
substituted explicitly, so callers know what they are getting:

  * `faceCaptureQuality`  -> approximated in `quality` by detection confidence
                             combined with face size and sharpness. It is a
                             different number from Vision's; do not port
                             macOS quality thresholds across verbatim.
  * yaw / roll / pitch    -> `pose()` estimates these from the five landmarks.
                             Roll is exact (an eye-line angle); yaw is a
                             reliable proxy from nose offset; pitch is coarse.

Post-processing follows InsightFace's reference SCRFD implementation:
three FPN strides (8/16/32), two anchors per location, distance-to-box and
distance-to-keypoint decoding, then NMS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

FEAT_STRIDES = (8, 16, 32)
NUM_ANCHORS = 2


@dataclass
class DetectedFace:
    """Pixel-space, top-left origin — same convention as the Swift `DetectedFace`."""

    box: np.ndarray        # (4,) x1, y1, x2, y2
    landmarks: np.ndarray  # (5, 2) left eye, right eye, nose, left mouth, right mouth (on-screen order)
    score: float
    quality: float

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    def pose(self) -> tuple[float, float, float]:
        """(yaw, roll, pitch) in radians, estimated from the five landmarks.

        Roll is exact. Yaw is derived from where the nose sits between the eyes:
        centred means facing forward, pushed toward one eye means turned away
        from it. Pitch compares the nose's height against the eye-to-mouth span
        and is the least trustworthy of the three.
        """
        le, re, nose, lm, rm = self.landmarks
        eye_vec = re - le
        roll = math.atan2(float(eye_vec[1]), float(eye_vec[0]))

        eye_mid = (le + re) / 2.0
        eye_dist = float(np.linalg.norm(eye_vec)) or 1.0
        # +1 when the nose sits over the right eye, -1 over the left.
        yaw_ratio = float(np.dot(nose - eye_mid, eye_vec / eye_dist)) / (eye_dist / 2.0)
        yaw = math.atan(max(-2.0, min(2.0, yaw_ratio)))

        mouth_mid = (lm + rm) / 2.0
        vertical = float(np.linalg.norm(mouth_mid - eye_mid)) or 1.0
        nose_frac = float(np.linalg.norm(nose - eye_mid)) / vertical
        # ~0.55 is a neutral, forward-facing nose position.
        pitch = math.atan((nose_frac - 0.55) * 2.0)
        return yaw, roll, pitch


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, 0] + distance[:, i])
        preds.append(points[:, 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= thresh)[0] + 1]
    return keep


class FaceDetector:
    def __init__(self, model_path, det_size: tuple[int, int] = (640, 640),
                 det_thresh: float = 0.5, nms_thresh: float = 0.4,
                 providers: list[str] | None = None) -> None:
        self.session = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self._anchor_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        ys, xs = np.mgrid[:height, :width]
        centers = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
        centers = centers.reshape(-1, 2)
        if NUM_ANCHORS > 1:
            centers = np.stack([centers] * NUM_ANCHORS, axis=1).reshape(-1, 2)
        self._anchor_cache[key] = centers
        return centers

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """`image_bgr` is an OpenCV BGR frame. Returns faces largest-first."""
        h0, w0 = image_bgr.shape[:2]
        target_w, target_h = self.det_size
        # Letterbox, preserving aspect ratio — SCRFD is sensitive to distortion.
        scale = min(target_w / w0, target_h / h0)
        new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, (target_w, target_h), (127.5, 127.5, 127.5), swapRB=True
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        fmc = len(FEAT_STRIDES)
        boxes_all, kps_all, scores_all = [], [], []
        for idx, stride in enumerate(FEAT_STRIDES):
            scores = outputs[idx]
            bbox_preds = outputs[idx + fmc] * stride
            kps_preds = outputs[idx + fmc * 2] * stride

            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            kps_preds = kps_preds.reshape(-1, 10)

            fh, fw = target_h // stride, target_w // stride
            centers = self._anchor_centers(fh, fw, stride)

            keep = np.where(scores >= self.det_thresh)[0]
            if keep.size == 0:
                continue
            boxes_all.append(_distance2bbox(centers[keep], bbox_preds[keep]))
            kps_all.append(_distance2kps(centers[keep], kps_preds[keep]).reshape(-1, 5, 2))
            scores_all.append(scores[keep])

        if not boxes_all:
            return []

        boxes = np.vstack(boxes_all) / scale
        kps = np.vstack(kps_all) / scale
        scores = np.concatenate(scores_all)

        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep = _nms(dets, self.nms_thresh)

        faces = []
        for i in keep:
            box = np.clip(boxes[i], [0, 0, 0, 0], [w0, h0, w0, h0])
            face = DetectedFace(
                box=box,
                landmarks=_order_landmarks(kps[i]),
                score=float(scores[i]),
                quality=0.0,
            )
            face.quality = _quality(face, image_bgr)
            faces.append(face)
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces


def _order_landmarks(kps: np.ndarray) -> np.ndarray:
    """SCRFD emits eyes first, then nose, then mouth corners — but which eye is
    'left' follows the model's convention, not the screen's. The macOS build hit
    the same trap with Vision's anatomical labels and fixed it by sorting on x;
    do the same so the ArcFace template always receives on-screen order.
    """
    kps = kps.astype(np.float32).copy()
    if kps[0, 0] > kps[1, 0]:
        kps[[0, 1]] = kps[[1, 0]]
    if kps[3, 0] > kps[4, 0]:
        kps[[3, 4]] = kps[[4, 3]]
    return kps


def _quality(face: DetectedFace, image_bgr: np.ndarray) -> float:
    """Stand-in for Vision's `faceCaptureQuality`, in 0..1.

    Blends detector confidence, how much of the frame the face fills, and
    Laplacian variance (sharpness) inside the box. It is a heuristic, and its
    numbers are NOT comparable to Vision's — thresholds must be tuned here.
    """
    x1, y1, x2, y2 = [int(v) for v in face.box]
    crop = image_bgr[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # ~150 is comfortably sharp for a webcam crop; saturate there.
    sharp_score = min(1.0, sharpness / 150.0)
    frame_area = image_bgr.shape[0] * image_bgr.shape[1]
    size_score = min(1.0, (face.area / frame_area) / 0.08)
    return float(0.5 * face.score + 0.3 * sharp_score + 0.2 * size_score)
