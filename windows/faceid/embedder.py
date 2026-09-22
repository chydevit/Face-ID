"""
ArcFace embedding — port of `ArcFaceEmbedder.swift` plus the `FaceEmbedding`
vector maths.

Runs InsightFace's w600k_mbf weights (MobileFaceNet backbone, ArcFace loss)
under onnxruntime. These are the same weights `tools/convert_arcface.py`
converts to Core ML for the macOS build, so embeddings are directly comparable
across the two platforms.

One deliberate difference: the Core ML package has preprocessing baked in, so
the Swift side hands over a raw 112x112 image. The raw ONNX graph does not, so
the (px - 127.5) / 127.5 scaling happens here instead.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

MODEL_IDENTIFIER = "arcface-w600k_mbf-v1"
EMBEDDING_DIMENSION = 512
INPUT_SIZE = 112


class ArcFaceEmbedder:
    name = "ArcFace (w600k_mbf)"
    model_identifier = MODEL_IDENTIFIER
    embedding_dimension = EMBEDDING_DIMENSION
    requires_alignment = True

    def __init__(self, model_path, providers: list[str] | None = None) -> None:
        self.session = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embedding(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """`aligned_bgr` must be the 112x112 output of `aligner.align`.
        Returns an L2-normalized float32 vector of length 512.
        """
        h, w = aligned_bgr.shape[:2]
        if (h, w) != (INPUT_SIZE, INPUT_SIZE):
            raise ValueError(
                f"ArcFaceEmbedder expects {INPUT_SIZE}x{INPUT_SIZE}, got {w}x{h}. "
                "Run the face through aligner.align first."
            )
        rgb = aligned_bgr[:, :, ::-1].astype(np.float32)
        blob = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None, ...]
        out = self.session.run([self.output_name], {self.input_name: blob})[0]
        return l2_normalized(np.asarray(out, dtype=np.float32).reshape(-1))


def l2_normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else (vector / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Range -1..1. The raw value ArcFace thresholds are quoted in."""
    if a.shape != b.shape or a.size == 0:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def average(vectors: list[np.ndarray]) -> np.ndarray | None:
    """Normalize each, mean, renormalize — a plain mean would let a
    larger-magnitude sample dominate. Port of `FaceEmbedding.average`.
    """
    if not vectors:
        return None
    dim = vectors[0].shape[0]
    usable = [l2_normalized(v) for v in vectors if v.shape[0] == dim]
    if not usable:
        return None
    return l2_normalized(np.mean(np.stack(usable), axis=0))
