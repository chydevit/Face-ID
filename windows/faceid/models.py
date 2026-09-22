"""
Model acquisition.

Both ONNX files come from InsightFace's official `buffalo_s` pack — the same
pack `tools/convert_arcface.py` feeds into the macOS build, so the recognition
weights here (`w600k_mbf`) are bit-identical to the ones Core ML runs on a Mac.
That is deliberate: an embedding enrolled on macOS and one computed on Windows
land in the same 512-dimensional space, so a future shared enrollment format
does not need a re-enroll.

Weights are downloaded on first run rather than vendored, for the same reason
the macOS tooling does it: ~13MB of third-party binary has no business in git.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"

# Members of the pack we actually use.
DETECTOR_MEMBER = "det_500m.onnx"
EMBEDDER_MEMBER = "w600k_mbf.onnx"


def model_dir() -> Path:
    """Per-user cache. %LOCALAPPDATA% on Windows, XDG-ish elsewhere."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "FaceID" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path, progress: bool = True) -> None:
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        last_pct = -1
        with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent) as tmp:
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                tmp.write(chunk)
                read += len(chunk)
                if progress and total:
                    pct = read * 100 // total
                    if pct != last_pct:      # only on change — otherwise this
                        last_pct = pct       # floods non-tty logs with one line per chunk
                        print(f"\r  downloading {dest.name}: {pct:3d}%", end="", flush=True)
        if progress:
            print()
        shutil.move(tmp.name, dest)


def ensure_models(progress: bool = True) -> tuple[Path, Path]:
    """Returns (detector_path, embedder_path), downloading the pack if needed."""
    d = model_dir()
    det, rec = d / DETECTOR_MEMBER, d / EMBEDDER_MEMBER
    if det.exists() and rec.exists():
        return det, rec

    if progress:
        print(f"Fetching InsightFace buffalo_s into {d}")
    archive = d / "buffalo_s.zip"
    try:
        _download(PACK_URL, archive, progress)
        with zipfile.ZipFile(archive) as z:
            for member in (DETECTOR_MEMBER, EMBEDDER_MEMBER):
                # Pack layout varies between releases (some nest under buffalo_s/).
                match = next((n for n in z.namelist() if n.endswith(member)), None)
                if match is None:
                    raise RuntimeError(f"{member} not found in buffalo_s.zip")
                with z.open(match) as src, open(d / member, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    finally:
        archive.unlink(missing_ok=True)

    if not (det.exists() and rec.exists()):
        raise RuntimeError("model extraction failed")
    return det, rec


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
