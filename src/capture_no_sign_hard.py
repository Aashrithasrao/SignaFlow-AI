from pathlib import Path
import cv2
import numpy as np

from src.camera_landmarks import CameraLandmarkExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUT_DIR = PROJECT_ROOT / "dataset" / "no_sign"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 32
TARGET_SAMPLES = 400
CAPTURE_EVERY = 10

extractor = CameraLandmarkExtractor()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

sequence = []
saved = 0
frame_count = 0

print("=" * 65)
print("SIGNAFLOW — HARD NO_SIGN COLLECTION")
print("=" * 65)
print()
print("We are collecting NON-SIGNING hand movements.")
print()
print("Do NOT make:")
print("  HELLO")
print("  MOM")
print("  DAD")
print("  WATER")
print("  FOOD")
print()
print("Instead do:")
print("  - hands visible but relaxed")
print("  - move hand casually")
print("  - put hand near face")
print("  - move hand across camera")
print("  - open/close hand casually")
print("  - change sitting position")
print("  - move hands between positions")
print("  - random gestures that are NOT the 5 signs")
print()
print("Target:", TARGET_SAMPLES)
print("Press Q to stop.")
print()

while saved < TARGET_SAMPLES:

    ok, frame = cap.read()

    if not ok:
        continue

    frame = cv2.flip(frame, 1)

    landmarks, result = extractor.extract(frame)

    sequence.append(landmarks)

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    frame_count += 1

    if (
        len(sequence) == SEQUENCE_LENGTH
        and frame_count % CAPTURE_EVERY == 0
    ):

        sample = np.asarray(
            sequence,
            dtype=np.float32
        )

        # Give every sample a unique name.
        output = OUT_DIR / f"hard_no_sign_{saved:04d}.npy"

        np.save(output, sample)

        saved += 1

        print(
            f"Captured {saved}/{TARGET_SAMPLES}",
            end="\r",
            flush=True
        )

    cv2.putText(
        frame,
        f"Hard NO_SIGN: {saved}/{TARGET_SAMPLES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Random movement - DO NOT SIGN",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "SignaFlow - Hard NO_SIGN",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
extractor.landmarker.close()
cv2.destroyAllWindows()

print()
print()
print(f"Finished: {saved} hard NO_SIGN samples.")
