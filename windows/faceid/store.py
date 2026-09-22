"""
Encrypted enrollment store.

The macOS build layers this over the Keychain: a Touch-ID-gated session key
wraps an ungated encrypted blob (`SecureCredentialManager.swift`). Windows has
no Keychain, so the equivalent is DPAPI — `CryptProtectData` ties the ciphertext
to the logged-in Windows user account, and no passphrase has to be stored or
prompted for.

HONEST LIMITATION: DPAPI protects against another user on the same machine
reading your enrollment. It does NOT protect against code running as you, since
that code can simply call `CryptUnprotectData` too. That matches what the
macOS build achieves once its session is unlocked, and is the same trade-off
Howdy makes on Linux. Do not treat this file as secret-proof.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .embedder import EMBEDDING_DIMENSION, MODEL_IDENTIFIER, average, cosine_similarity

IS_WINDOWS = sys.platform == "win32"


def data_dir() -> Path:
    if IS_WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    d = base / "FaceID"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- DPAPI -----------------------------------------------------------------

class _DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes) -> _DATA_BLOB:
    buf = ctypes.create_string_buffer(data, len(data))
    return _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))


def _blob_bytes(blob: _DATA_BLOB) -> bytes:
    return ctypes.string_at(blob.pbData, blob.cbData)


def protect(plaintext: bytes) -> bytes:
    if not IS_WINDOWS:
        return plaintext
    out = _DATA_BLOB()
    ok = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(_blob(plaintext)), "FaceID enrollment", None, None, None, 0, ctypes.byref(out)
    )
    if not ok:
        raise OSError("CryptProtectData failed")
    try:
        return _blob_bytes(out)
    finally:
        ctypes.windll.kernel32.LocalFree(out.pbData)


def unprotect(ciphertext: bytes) -> bytes:
    if not IS_WINDOWS:
        return ciphertext
    out = _DATA_BLOB()
    ok = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(_blob(ciphertext)), None, None, None, None, 0, ctypes.byref(out)
    )
    if not ok:
        raise OSError("CryptUnprotectData failed — enrollment belongs to another Windows user")
    try:
        return _blob_bytes(out)
    finally:
        ctypes.windll.kernel32.LocalFree(out.pbData)


# --- Records ---------------------------------------------------------------

@dataclass
class FaceSample:
    embedding: list
    pose: str | None
    quality: float | None
    captured_at: str


@dataclass
class FaceIdentity:
    """Mirrors `FaceIdentity` in the Swift build, including the model guard:
    embeddings from different models must never be compared — that produces
    confident nonsense rather than an error.
    """

    id: str
    name: str
    samples: list = field(default_factory=list)
    model_identifier: str = MODEL_IDENTIFIER
    embedding_dimension: int = EMBEDDING_DIMENSION
    created_at: str = ""
    enabled: bool = True

    def template(self) -> np.ndarray | None:
        vectors = [np.asarray(s["embedding"], dtype=np.float32) for s in self.samples]
        return average(vectors)


class EnrollmentStore:
    FILENAME = "enrollment.bin"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (data_dir() / self.FILENAME)
        self.identities: list[FaceIdentity] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.identities = []
            return
        raw = unprotect(self.path.read_bytes())
        payload = json.loads(raw.decode("utf-8"))
        self.identities = [FaceIdentity(**rec) for rec in payload.get("identities", [])]

    def save(self) -> None:
        payload = {"version": 1, "identities": [asdict(i) for i in self.identities]}
        blob = protect(json.dumps(payload).encode("utf-8"))
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, self.path)
        if not IS_WINDOWS:
            os.chmod(self.path, 0o600)

    def add(self, name: str, embeddings: list, qualities: list | None = None,
            poses: list | None = None) -> FaceIdentity:
        now = datetime.now(timezone.utc).isoformat()
        samples = []
        for i, v in enumerate(embeddings):
            samples.append({
                "embedding": [float(x) for x in np.asarray(v).reshape(-1)],
                "pose": (poses[i] if poses and i < len(poses) else None),
                "quality": (float(qualities[i]) if qualities and i < len(qualities) else None),
                "captured_at": now,
            })
        identity = FaceIdentity(
            id=str(uuid.uuid4()), name=name, samples=samples, created_at=now
        )
        # Re-enrolling under an existing name replaces it rather than stacking.
        self.identities = [i for i in self.identities if i.name != name]
        self.identities.append(identity)
        self.save()
        return identity

    def remove(self, name: str) -> bool:
        before = len(self.identities)
        self.identities = [i for i in self.identities if i.name != name]
        if len(self.identities) != before:
            self.save()
            return True
        return False

    def best_match(self, probe: np.ndarray, threshold: float) -> tuple[FaceIdentity | None, float]:
        """Port of `FaceRecognitionPipeline.bestMatch`: BOTH the centroid and the
        single best sample must clear the threshold, so one lucky frame cannot
        carry a match on its own.
        """
        best, best_score = None, -1.0
        for identity in self.identities:
            if not identity.enabled:
                continue
            if identity.model_identifier != MODEL_IDENTIFIER:
                continue  # never compare across embedders
            template = identity.template()
            if template is None:
                continue
            centroid_sim = cosine_similarity(probe, template)
            sample_sims = [
                cosine_similarity(probe, np.asarray(s["embedding"], dtype=np.float32))
                for s in identity.samples
            ]
            max_sim = max(sample_sims) if sample_sims else -1.0
            score = min(centroid_sim, max_sim)
            if score > best_score:
                best, best_score = identity, score
        if best is not None and best_score >= threshold:
            return best, best_score
        return None, best_score
