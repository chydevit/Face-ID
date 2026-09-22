"""
Command line for the Windows face recognition engine.

    faceid enroll --name Alice     capture samples and store them
    faceid verify                  live match against the store
    faceid list / forget           manage enrollments
    faceid selftest                offline check that the pipeline is sane
    faceid cameras                 probe camera indices

SCOPE — be clear about this. These commands RECOGNIZE a face. They do not log
you into Windows. Replacing the Windows logon needs a COM Credential Provider
DLL loaded by LogonUI, which is a separate C++ component; see windows/README.md.
The engine here is what that component will call.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# Same default as the macOS build's `GlanceSettings.matchThreshold` (0.66).
# ArcFace cosine for genuine pairs typically lands far above this; 0.66 is
# deliberately strict, favouring false rejects over false accepts.
DEFAULT_THRESHOLD = 0.66


def _engine(providers=None):
    from .models import ensure_models
    from .detector import FaceDetector
    from .embedder import ArcFaceEmbedder

    det_path, rec_path = ensure_models()
    return FaceDetector(det_path, providers=providers), ArcFaceEmbedder(rec_path, providers=providers)


def cmd_enroll(args) -> int:
    from .aligner import align
    from .camera import Camera
    from .store import EnrollmentStore

    detector, embedder = _engine()
    store = EnrollmentStore()

    print(f"Enrolling '{args.name}'. Look at the camera and move your head slowly.")
    print(f"Capturing {args.samples} samples; press Ctrl+C to abort.\n")

    embeddings, qualities = [], []
    with Camera(args.camera) as cam:
        last = 0.0
        while len(embeddings) < args.samples:
            frame = cam.read()
            if frame is None:
                continue
            faces = detector.detect(frame)
            if not faces:
                continue
            face = faces[0]
            if face.quality < args.min_quality:
                continue
            if time.time() - last < args.interval:
                continue
            aligned = align(frame, face.landmarks, face.box)
            if aligned is None or aligned.tier != "5-point":
                continue
            embeddings.append(embedder.embedding(aligned.image))
            qualities.append(face.quality)
            last = time.time()
            print(f"  captured {len(embeddings)}/{args.samples}  quality={face.quality:.2f}")

    identity = store.add(args.name, embeddings, qualities)
    print(f"\nStored {len(identity.samples)} samples for '{args.name}' in {store.path}")

    from .embedder import cosine_similarity
    template = identity.template()
    sims = [cosine_similarity(np.asarray(s["embedding"], dtype=np.float32), template)
            for s in identity.samples]
    print(f"Sample-to-template similarity: min={min(sims):.3f} mean={sum(sims)/len(sims):.3f}")
    if min(sims) < 0.5:
        print("WARNING: samples disagree a lot. Re-enroll in even lighting.")
    return 0


def cmd_verify(args) -> int:
    from .aligner import align
    from .camera import Camera
    from .liveness import LivenessAnalyzer, HEAVY, LIGHT
    from .store import EnrollmentStore

    detector, embedder = _engine()
    store = EnrollmentStore()
    if not store.identities:
        print("Nothing enrolled. Run: faceid enroll --name <you>")
        return 2

    analyzer = LivenessAnalyzer(mode=HEAVY if args.heavy else LIGHT)
    deadline = time.time() + args.timeout
    print(f"Looking for a match (threshold {args.threshold}, "
          f"liveness {'heavy' if args.heavy else 'light'}, {args.timeout}s)...")

    best_seen = -1.0
    with Camera(args.camera) as cam:
        while time.time() < deadline:
            frame = cam.read()
            if frame is None:
                continue
            faces = detector.detect(frame)
            if not faces:
                continue
            face = faces[0]
            analyzer.observe(frame, face)
            aligned = align(frame, face.landmarks, face.box)
            if aligned is None:
                continue
            probe = embedder.embedding(aligned.image)
            identity, score = store.best_match(probe, args.threshold)
            best_seen = max(best_seen, score)

            if identity is None:
                continue
            snap = analyzer.snapshot()
            if not snap.passed:
                print(f"  match {identity.name} ({score:.3f}) REJECTED by liveness:")
                for r in snap.reasons:
                    print(f"    - {r}")
                continue
            print(f"\nMATCH: {identity.name}  similarity={score:.3f}")
            for r in snap.reasons:
                print(f"  liveness: {r}")
            return 0

    print(f"\nNo match. Best similarity seen: {best_seen:.3f} (threshold {args.threshold})")
    return 1


def cmd_list(args) -> int:
    from .store import EnrollmentStore

    store = EnrollmentStore()
    if not store.identities:
        print("No enrollments.")
        return 0
    for i in store.identities:
        state = "enabled" if i.enabled else "disabled"
        print(f"{i.name:20} {len(i.samples):3d} samples  {i.model_identifier}  {state}  {i.created_at}")
    return 0


def cmd_forget(args) -> int:
    from .store import EnrollmentStore

    store = EnrollmentStore()
    print(f"Removed '{args.name}'" if store.remove(args.name) else f"No enrollment named '{args.name}'")
    return 0


def cmd_cameras(args) -> int:
    from .camera import list_cameras

    found = list_cameras()
    print("Working camera indices:", found or "none found")
    return 0 if found else 1


def cmd_selftest(args) -> int:
    """Offline sanity check — no camera needed.

    Verifies the same property the numbers in windows/README.md report: a face
    and a transformed copy of it must score far above threshold, and two
    different faces far below.
    """
    import cv2
    from .aligner import align
    from .embedder import cosine_similarity

    if not args.images or len(args.images) < 1:
        print("Pass at least one face image: faceid selftest a.jpg [b.jpg]")
        return 2

    detector, embedder = _engine()

    def embed(path):
        img = cv2.imread(path)
        if img is None:
            print(f"  cannot read {path}")
            return None
        faces = detector.detect(img)
        if not faces:
            print(f"  no face found in {path}")
            return None
        f = faces[0]
        a = align(img, f.landmarks, f.box)
        print(f"  {path}: score={f.score:.3f} quality={f.quality:.3f} tier={a.tier}")
        return img, embedder.embedding(a.image)

    first = embed(args.images[0])
    if first is None:
        return 1
    img, v1 = first

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 0.85)
    warped = cv2.convertScaleAbs(cv2.warpAffine(img, M, (w, h)), alpha=1.15, beta=12)
    faces = detector.detect(warped)
    if not faces:
        print("  transformed copy lost the face — detector regression?")
        return 1
    f = faces[0]
    v2 = embedder.embedding(align(warped, f.landmarks, f.box).image)
    same = cosine_similarity(v1, v2)
    print(f"\n  same face, transformed : {same:+.4f}  (want > {DEFAULT_THRESHOLD})")

    ok = same > DEFAULT_THRESHOLD
    if len(args.images) > 1:
        other = embed(args.images[1])
        if other is not None:
            cross = cosine_similarity(v1, other[1])
            print(f"  different faces        : {cross:+.4f}  (want < 0.30)")
            ok = ok and cross < 0.30

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="faceid", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("enroll", help="capture and store face samples")
    p.add_argument("--name", required=True)
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--interval", type=float, default=0.4, help="seconds between captures")
    p.add_argument("--min-quality", type=float, default=0.45)
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("verify", help="live match against the store")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--heavy", action="store_true", help="require a positive liveness cue")
    p.set_defaults(func=cmd_verify)

    sub.add_parser("list", help="show enrollments").set_defaults(func=cmd_list)

    p = sub.add_parser("forget", help="remove an enrollment")
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_forget)

    sub.add_parser("cameras", help="probe camera indices").set_defaults(func=cmd_cameras)

    p = sub.add_parser("selftest", help="offline pipeline check")
    p.add_argument("images", nargs="*")
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\naborted")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1
