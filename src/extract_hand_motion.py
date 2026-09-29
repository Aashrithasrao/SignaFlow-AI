import cv2
import numpy as np
from pathlib import Path
import sys

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VIDEO = (
    PROJECT_ROOT
    / "signs"
    / "water"
    / "water_clean.mp4"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

OUTPUT_DIR = PROJECT_ROOT / "sign_motion"

NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 3


def create_landmarker():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hand Landmarker model not found:\n{MODEL_PATH}"
        )

    base_options = BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )

    return vision.HandLandmarker.create_from_options(options)


def extract(video_path: Path):

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open:\n{video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print("=" * 70)
    print("SIGNAFLOW — DEDICATED HAND LANDMARKER")
    print("=" * 70)
    print(f"Video  : {video_path}")
    print(f"FPS    : {fps:.2f}")
    print(f"Frames : {total_frames}")
    print()
    print("Detector settings:")
    print("  Hands                 : 2")
    print("  Detection confidence  : 0.25")
    print("  Presence confidence   : 0.25")
    print("  Tracking confidence   : 0.25")
    print()

    landmarker = create_landmarker()

    all_frames = []

    frames_with_hands = 0
    frames_with_left = 0
    frames_with_right = 0

    frame_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        # Enlarge the source before detection.
        enlarged = cv2.resize(
            frame,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )

        rgb = cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = landmarker.detect(image)

        # --------------------------------------------------
        # Store detected hands as:
        #
        # [left 21x3, right 21x3]
        #
        # Missing hand = zeros.
        # --------------------------------------------------

        left = np.zeros(
            (21, 3),
            dtype=np.float32
        )

        right = np.zeros(
            (21, 3),
            dtype=np.float32
        )

        left_found = False
        right_found = False

        for hand_landmarks, handedness in zip(
            result.hand_landmarks,
            result.handedness,
        ):

            if not handedness:
                continue

            label = handedness[0].category_name

            points = np.array(
                [
                    [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                    ]
                    for landmark in hand_landmarks
                ],
                dtype=np.float32,
            )

            if label.lower() == "left":
                left = points
                left_found = True

            elif label.lower() == "right":
                right = points
                right_found = True

        if left_found:
            frames_with_left += 1

        if right_found:
            frames_with_right += 1

        if left_found or right_found:
            frames_with_hands += 1

        combined = np.concatenate(
            [
                left.flatten(),
                right.flatten(),
            ]
        )

        all_frames.append(combined)

        frame_index += 1

        print(
            f"\rProcessing "
            f"{frame_index}/{total_frames}",
            end="",
            flush=True,
        )

    cap.release()
    landmarker.close()

    print()
    print()

    motion = np.asarray(
        all_frames,
        dtype=np.float32
    )

    output_path = (
        OUTPUT_DIR
        / f"{video_path.stem}_hands_126.npy"
    )

    np.save(
        output_path,
        motion
    )

    print("=" * 70)
    print("HAND EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Motion shape : {motion.shape}")
    print(f"Saved to     : {output_path}")
    print()
    print(
        f"Any hand : "
        f"{frames_with_hands}/{total_frames}"
    )
    print(
        f"Left hand: "
        f"{frames_with_left}/{total_frames}"
    )
    print(
        f"Right hand: "
        f"{frames_with_right}/{total_frames}"
    )
    print()

    coverage = (
        frames_with_hands / total_frames
        if total_frames
        else 0
    )

    if coverage >= 0.90:
        print(
            "EXCELLENT: hand coverage is "
            "strong enough for the next stage."
        )
    elif coverage >= 0.70:
        print(
            "GOOD: hand coverage is usable, "
            "but interpolation will be needed."
        )
    else:
        print(
            "INSUFFICIENT: the source video "
            "needs a different tracking strategy."
        )

    print("=" * 70)


if __name__ == "__main__":

    video = (
        Path(sys.argv[1]).expanduser()
        if len(sys.argv) > 1
        else DEFAULT_VIDEO
    )

    extract(video)
