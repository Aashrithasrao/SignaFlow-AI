from pathlib import Path
import cv2
import numpy as np

from src.camera_landmarks import CameraLandmarkExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUT_DIR = PROJECT_ROOT / "dataset" / "no_sign"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 32
TARGET_SAMPLES = 250
CAPTURE_EVERY = 15

extractor = CameraLandmarkExtractor()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

sequence = []
saved = 0
frame_count = 0

print("=" * 60)
print("NO_SIGN DATA COLLECTION")
print("=" * 60)
print()
print("Sit normally in front of the camera.")
print("Keep your hands relaxed/down.")
print("Make small natural movements.")
print()
print("Do NOT perform any of the 5 signs.")
print()
print("Press Q to stop.")
print()
print(f"Target samples: {TARGET_SAMPLES}")
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

        output = OUT_DIR / f"no_sign_{saved:04d}.npy"

        np.save(output, sample)

        saved += 1

        print(
            f"Captured {saved}/{TARGET_SAMPLES}",
            end="\r",
            flush=True
        )

    cv2.putText(
        frame,
        f"NO_SIGN samples: {saved}/{TARGET_SAMPLES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Stay relaxed - do NOT sign",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "SignaFlow - NO_SIGN Collection",
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
print(f"Finished. Captured {saved} NO_SIGN samples.")
print(f"Saved to: {OUT_DIR}")
