import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera_landmarks import CameraLandmarkExtractor
from src.preprocess_landmarks import normalize_sequence

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "camera_no" / "no"
TARGET_SAMPLES = 50
SEQUENCE_LENGTH = 32


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_samples = sorted(OUTPUT_DIR.glob("*.npy"))
    count = len(existing_samples)

    print("=" * 70)
    print("SIGNAFLOW AI — NO SIGN REAL-WEBCAM DATA COLLECTOR")
    print("=" * 70)
    print(f"Existing samples in {OUTPUT_DIR.name}: {count}/{TARGET_SAMPLES}")
    print("\nControls:")
    print("  [SPACE] or [N] : Start recording one 'NO' sample (with 1.5s countdown)")
    print("  [Q]            : Save & Exit")
    print("=" * 70 + "\n")

    extractor = CameraLandmarkExtractor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording = False
    countdown_start = None
    COUNTDOWN_DURATION = 1.2  # seconds before capture begins
    current_sequence = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        features, result = extractor.extract(frame)
        hands_present = bool(result.left_hand_landmarks or result.right_hand_landmarks)

        # Draw header overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (18, 22, 32), -1)
        cv2.rectangle(overlay, (0, h - 70), (w, h), (18, 22, 32), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        now = time.time()

        if countdown_start is not None:
            elapsed = now - countdown_start
            remaining = COUNTDOWN_DURATION - elapsed
            if remaining > 0:
                cv2.putText(
                    frame,
                    f"GET READY: {remaining:.1f}s",
                    (w // 2 - 160, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 230, 255),
                    3,
                    cv2.LINE_AA,
                )
                status_text = "GET READY... Position hand at chest level"
                status_color = (0, 230, 255)
            else:
                countdown_start = None
                recording = True
                current_sequence = []
                status_text = "SIGN 'NO' NOW (TAP INDEX + MIDDLE TO THUMB TWICE)!"
                status_color = (0, 255, 120)
        elif recording:
            current_sequence.append(features)
            status_text = f"RECORDING 'NO'... [{len(current_sequence)}/{SEQUENCE_LENGTH} frames]"
            status_color = (0, 255, 120)

            # Draw flashing recording dot
            cv2.circle(frame, (w - 40, 45), 14, (0, 0, 255), -1)

            if len(current_sequence) >= SEQUENCE_LENGTH:
                # Save normalized sequence
                seq_arr = np.asarray(current_sequence[:SEQUENCE_LENGTH], dtype=np.float32)
                seq_norm = normalize_sequence(seq_arr)

                count += 1
                out_path = OUTPUT_DIR / f"no_camera_{count:04d}.npy"
                np.save(out_path, seq_norm)
                print(f"[{count}/{TARGET_SAMPLES}] Saved {out_path.name}")

                recording = False
                current_sequence.clear()
        else:
            status_text = "PRESS [SPACE] OR [N] TO RECORD 'NO' SAMPLE"
            status_color = (255, 255, 255)

        # Title
        cv2.putText(
            frame,
            f"SIGNAFLOW — 'NO' CAMERA SAMPLE RECORDER  |  COLLECTED: {count}/{TARGET_SAMPLES}",
            (25, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 230, 255),
            2,
            cv2.LINE_AA,
        )

        # Status text
        cv2.putText(
            frame,
            status_text,
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # Footer guide
        guide_text = "HOW TO SIGN 'NO': Dominant hand at chest height, tap index+middle tips against thumb twice"
        cv2.putText(
            frame,
            guide_text,
            (25, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 210, 220),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("SignaFlow — Collect NO Sign Samples", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), 27]:
            print("\nExiting collection...")
            break
        elif key in [ord(" "), ord("n"), ord("N")]:
            if not recording and countdown_start is None:
                countdown_start = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCollection session finished. Total samples saved in {OUTPUT_DIR}: {count}")


if __name__ == "__main__":
    main()
