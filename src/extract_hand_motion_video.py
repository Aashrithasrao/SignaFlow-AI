import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "signs" / "water" / "water_clean.mp4"
HAND_MODEL = PROJECT_ROOT / "models" / "mediapipe" / "hand_landmarker.task"
OUTPUT_DIR = PROJECT_ROOT / "sign_motion"

# Try to detect hands even when the source is soft/blurry.
DETECTION_CONFIDENCE = 0.10
PRESENCE_CONFIDENCE = 0.10
TRACKING_CONFIDENCE = 0.10


def create_landmarker():

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(HAND_MODEL)
        ),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=PRESENCE_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    return vision.HandLandmarker.create_from_options(options)


def interpolate_missing(data):

    """
    Fill missing hand frames using temporal interpolation.

    data:
        (frames, 21, 3)
    """

    result = data.copy()

    frames = len(result)

    for landmark in range(21):

        for axis in range(3):

            values = result[:, landmark, axis]

            valid = np.any(
                np.abs(result[:, landmark, :]) > 1e-8,
                axis=1,
            )

            if valid.sum() < 2:
                continue

            valid_indices = np.where(valid)[0]

            result[:, landmark, axis] = np.interp(
                np.arange(frames),
                valid_indices,
                values[valid],
            )

    return result


def extract():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(VIDEO_PATH)

    if not HAND_MODEL.exists():
        raise FileNotFoundError(HAND_MODEL)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open WATER video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 70)
    print("SIGNAFLOW — TEMPORAL HAND TRACKING")
    print("=" * 70)
    print(f"Video : {VIDEO_PATH}")
    print(f"FPS   : {fps:.2f}")
    print(f"Frames: {total}")
    print()
    print("MediaPipe VIDEO mode")
    print("Temporal tracking enabled")
    print()

    landmarker = create_landmarker()

    left_frames = []
    right_frames = []

    left_detected = 0
    right_detected = 0
    any_detected = 0

    frame_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        # --------------------------------------------------
        # Mild enlargement.
        # --------------------------------------------------

        enlarged = cv2.resize(
            frame,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )

        rgb = cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2RGB,
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # Timestamp MUST increase monotonically.
        timestamp_ms = int(
            frame_index * 1000 / fps
        )

        result = landmarker.detect_for_video(
            image,
            timestamp_ms,
        )

        left = np.zeros(
            (21, 3),
            dtype=np.float32,
        )

        right = np.zeros(
            (21, 3),
            dtype=np.float32,
        )

        if result.hand_landmarks:

            for hand_index, hand in enumerate(
                result.hand_landmarks
            ):

                if hand_index >= len(
                    result.handedness
                ):
                    continue

                handedness = (
                    result.handedness[hand_index][0]
                )

                label = handedness.category_name.lower()

                points = np.array(
                    [
                        [p.x, p.y, p.z]
                        for p in hand
                    ],
                    dtype=np.float32,
                )

                # MediaPipe reports the hand from
                # the camera's perspective.
                if label == "left":
                    left = points
                    left_detected += 1

                elif label == "right":
                    right = points
                    right_detected += 1

        if (
            np.any(left)
            or np.any(right)
        ):
            any_detected += 1

        left_frames.append(left)
        right_frames.append(right)

        frame_index += 1

        print(
            f"\rProcessing {frame_index}/{total}",
            end="",
            flush=True,
        )

    cap.release()
    landmarker.close()

    left_frames = np.asarray(
        left_frames,
        dtype=np.float32,
    )

    right_frames = np.asarray(
        right_frames,
        dtype=np.float32,
    )

    print()
    print()

    # ------------------------------------------------------
    # Interpolate temporal gaps.
    # ------------------------------------------------------

    print("Interpolating missing frames...")

    left_filled = interpolate_missing(left_frames)
    right_filled = interpolate_missing(right_frames)

    combined = np.concatenate(
        [
            left_filled.reshape(total, 63),
            right_filled.reshape(total, 63),
        ],
        axis=1,
    )

    output_path = (
        OUTPUT_DIR
        / "water_temporal_hands_126.npy"
    )

    np.save(
        output_path,
        combined,
    )

    print("=" * 70)
    print("TEMPORAL EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Raw left detected  : "
        f"{left_detected}/{total}"
    )

    print(
        f"Raw right detected : "
        f"{right_detected}/{total}"
    )

    print(
        f"Raw any detected   : "
        f"{any_detected}/{total}"
    )

    print(
        f"Motion shape       : "
        f"{combined.shape}"
    )

    print(
        f"Saved              : "
        f"{output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    extract()
