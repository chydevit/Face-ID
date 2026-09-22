# Face ID for Windows

A Windows face-recognition engine that reuses the macOS build's recognition
core. It is **not** a port of the Mac app — none of that Swift runs here. What
carries over is the part that is genuinely portable.

## What is shared with the macOS build

| Piece | Shared? |
|---|---|
| ArcFace weights (`w600k_mbf`, 512-d) | **Identical.** Same InsightFace `buffalo_s` pack `tools/convert_arcface.py` converts to Core ML. Embeddings are directly comparable across the two platforms. |
| 112×112 alignment template | **Identical constants**, ported from `Face ID/FaceAligner.swift`. |
| Cosine matching, template averaging | Ported from `FaceEmbedding`. |
| `bestMatch` rule (centroid **and** best sample must clear threshold) | Ported from `FaceRecognitionPipeline.swift`. |
| Face detection | **Replaced.** Vision is Apple-only; SCRFD (`det_500m.onnx`) stands in. |
| Capture quality, yaw/roll/pitch | **Replaced with heuristics.** Vision gave these free; see `detector.py`. Numbers are not comparable — do not carry macOS thresholds across. |
| Credential storage | **Replaced.** Keychain + Touch ID → DPAPI. |
| Liveness | **Partial.** See below. |
| Notch UI, lock-screen unlock | **Absent.** See "What this does not do". |

## What this does not do

**It does not log you into Windows.** It recognizes a face and exits with a
status code. Replacing the Windows logon requires a COM **Credential Provider**
DLL (`ICredentialProvider`) loaded by LogonUI — a separate C++ component that
this engine would be called by.

That is also why the macOS approach could not simply be carried over. Glance on
macOS types your stored password at the lock screen, because macOS gives
third-party apps no way to authorize a login. On Windows that trick is blocked
by design: LogonUI runs on the separate Winlogon secure desktop, and a process
in your session cannot `SendInput` into it. The Credential Provider route is
the supported path — and a better one, because **it never stores your password
at all**.

## Liveness is weaker here than on macOS

The macOS build runs five cues. This one runs four, two of them reduced:

| Cue | Status |
|---|---|
| Gloss / glare | Reimplemented (pure pixel work, transfers cleanly) |
| Device detected | Reimplemented (rectangle enclosing the face) |
| Depth / pose | Ported — needs only the 5 points SCRFD gives |
| Motion | New, partially replacing flat-vs-3D |
| **Blink** | **Missing.** Eye-aspect-ratio needs per-eyelid contours; SCRFD emits one point per eye. Restoring it needs InsightFace's `2d106det`. |
| **Flat vs 3D** | **Partial.** The macOS plane fit needs dense landmarks. |

**This is measurably easier to spoof than the macOS build**, and its thresholds
are first-pass values chosen by inspection, not tuned against a spoof corpus.
Treat it as a starting point.

## Install

Download `faceid.exe` from the [releases page](https://github.com/chydevit/Face-ID/releases),
or run from source:

```bash
cd windows
pip install -r requirements.txt
python -m faceid selftest
```

Models (~16 MB) download from InsightFace on first run into
`%LOCALAPPDATA%\FaceID\models`.

## Use

```bash
faceid.exe cameras                      # probe camera indices
faceid.exe enroll --name Alice          # capture 12 samples
faceid.exe list                         # show enrollments
faceid.exe verify                       # live match, exit 0 on success
faceid.exe verify --heavy               # require a positive liveness cue
faceid.exe forget --name Alice
faceid.exe selftest a.jpg b.jpg         # offline pipeline check, no camera
```

`verify` exits `0` on match, `1` on no match, `2` if nothing is enrolled — so it
composes into scripts.

## Verification

The pipeline was checked end to end before release. On the OpenCV sample faces:

```
same face (rotated 8°, scaled 0.85, brightened) : +0.9577
two different faces                             : +0.0323
enrolled template vs. same face                 : +0.9908
enrolled template vs. different face            : +0.0301  (no match)
```

Threshold is `0.66`, matching `GlanceSettings.matchThreshold` in the macOS
build. The CI job reruns `selftest` on a Windows runner against both the source
and the packaged `.exe`, so a published binary has been shown to work on the
platform it ships for.

**Not yet verified:** live camera capture on Windows, DPAPI round-trip on
Windows (the store was exercised on the non-Windows fallback path), and
end-to-end enroll/verify with a real person. Those need a Windows machine with
a webcam — a CI runner has neither.

## Where this goes next

1. Add `2d106det` for dense landmarks — restores blink and the real flat-vs-3D plane fit.
2. Wrap the engine in a long-lived local service so the logon path does not pay model-load cost per attempt.
3. Write the Credential Provider DLL (C++/COM) that calls it. This is the large piece: ~6–10 weeks, and the only part that actually unlocks Windows.

## Security

Same fundamental caveat as the Mac build: a webcam sees a flat 2D image, not a
depth map. It defeats casual photo attacks and does not reliably defeat a video
of you. DPAPI ties the enrollment to your Windows account, which stops another
user on the machine reading it — it does **not** stop code running as you.

If your machine has an IR camera, **Windows Hello already does this natively
and far better.** This is aimed at machines without Hello-capable hardware.
