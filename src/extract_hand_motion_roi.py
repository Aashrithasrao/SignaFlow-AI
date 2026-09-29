import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

from camera_landmarks import CameraLandmarkExtractor
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    PROJECT_ROOT
    / "signs"
    / "water"
    / "water_clean.mp4"
)

HAND_MODEL = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

OUTPUT_DIR = PROJECT_ROOT / "sign_motion"

# MediaPipe pose wrist indices
LEFT_WRIST = 15
RIGHT_WRIST = 16

HAND_SIZE_MULTIPLIER = 2.2
UPSCALE = 4.0


def create_hand_landmarker():

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(HAND_MODEL)
        ),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.15,
        min_hand_presence_confidence=0.15,
        min_tracking_confidence=0.15,
    )

    return vision.HandLandmarker.create_from_options(
        options
    )


def crop_around_wrist(
    frame,
    wrist_x,
    wrist_y,
    crop_size,
):

    h, w = frame.shape[:2]

    cx = int(wrist_x * w)
    cy = int(wrist_y * h)

    half = int(crop_size / 2)

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)

    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def get_pose_point(result, index):

    if not result.pose_landmarks:
        return None

    if index >= len(result.pose_landmarks):
        return None

    landmark = result.pose_landmarks[index]

    # Reject obviously invalid pose points.
    if not np.isfinite(landmark.x):
        return None

    if not np.isfinite(landmark.y):
        return None

    return landmark


def hand_from_crop(
    landmarker,
    crop,
):

    if crop is None:
        return None

    enlarged = cv2.resize(
        crop,
        None,
        fx=UPSCALE,
        fy=UPSCALE,
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

    result = landmarker.detect(image)

    if not result.hand_landmarks:
        return None

    points = np.array(
        [
            [
                p.x,
                p.y,
                p.z,
            ]
            for p in result.hand_landmarks[0]
        ],
        dtype=np.float32,
    )

    return points


def extract():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    if not HAND_MODEL.exists():
        raise FileNotFoundError(
            f"Hand model not found:\n{HAND_MODEL}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open WATER video."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print("=" * 70)
    print("SIGNAFLOW — WRIST GUIDED HAND TRACKING")
    print("=" * 70)
    print(f"Video : {VIDEO_PATH}")
    print(f"FPS   : {fps:.2f}")
    print(f"Frames: {total}")
    print()
    print("Strategy:")
    print("  1. MediaPipe pose finds wrists")
    print("  2. Crop around each wrist")
    print("  3. 4× enlarge crop")
    print("  4. Dedicated hand detector")
    print()

    pose_extractor = CameraLandmarkExtractor()
    hand_landmarker = create_hand_landmarker()

    left_all = []
    right_all = []

    left_detected = 0
    right_detected = 0
    any_detected = 0

    frame_no = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        _, pose_result = pose_extractor.extract(
            frame
        )

        h, w = frame.shape[:2]

        # --------------------------------------------------
        # Estimate hand crop size from shoulder distance.
        # --------------------------------------------------

        left_shoulder = get_pose_point(
            pose_result,
            11,
        )

        right_shoulder = get_pose_point(
            pose_result,
            12,
        )

        if (
            left_shoulder is not None
            and right_shoulder is not None
        ):

            shoulder_distance = (
                abs(
                    left_shoulder.x
                    - right_shoulder.x
                )
                * w
            )

            crop_size = max(
                80,
                int(
                    shoulder_distance
                    * HAND_SIZE_MULTIPLIER
                ),
            )

        else:
            crop_size = 180

        # --------------------------------------------------
        # Get wrists.
        # --------------------------------------------------

        left_wrist = get_pose_point(
            pose_result,
            LEFT_WRIST,
        )

        right_wrist = get_pose_point(
            pose_result,
            RIGHT_WRIST,
        )

        left_points = np.zeros(
            (21, 3),
            dtype=np.float32,
        )

        right_points = np.zeros(
            (21, 3),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # LEFT HAND
        # --------------------------------------------------

        if left_wrist is not None:

            crop = crop_around_wrist(
                frame,
                left_wrist.x,
                left_wrist.y,
                crop_size,
            )

            points = hand_from_crop(
                hand_landmarker,
                crop,
            )

            if points is not None:

                left_points = points
                left_detected += 1

        # --------------------------------------------------
        # RIGHT HAND
        # --------------------------------------------------

        if right_wrist is not None:

            crop = crop_around_wrist(
                frame,
                right_wrist.x,
                right_wrist.y,
                crop_size,
            )

            points = hand_from_crop(
                hand_landmarker,
                crop,
            )

            if points is not None:

                right_points = points
                right_detected += 1

        if (
            np.any(left_points)
            or np.any(right_points)
        ):
            any_detected += 1

        left_all.append(left_points)
        right_all.append(right_points)

        frame_no += 1

        print(
            f"\rProcessing "
            f"{frame_no}/{total}",
            end="",
            flush=True,
        )

    cap.release()
    hand_landmarker.close()

    left_all = np.asarray(
        left_all,
        dtype=np.float32,
    )

    right_all = np.asarray(
        right_all,
        dtype=np.float32,
    )

    combined = np.concatenate(
        [
            left_all.reshape(total, -1),
            right_all.reshape(total, -1),
        ],
        axis=1,
    )

    output_path = (
        OUTPUT_DIR
        / "water_roi_hands_126.npy"
    )

    np.save(
        output_path,
        combined,
    )

    print()
    print()
    print("=" * 70)
    print("WRIST-GUIDED EXTRACTION COMPLETE")
    print("=" * 70)
    print(
        f"Any hand detected : "
        f"{any_detected}/{total}"
    )
    print(
        f"Left hand         : "
        f"{left_detected}/{total}"
    )
    print(
        f"Right hand        : "
        f"{right_detected}/{total}"
    )
    print(
        f"Motion shape      : "
        f"{combined.shape}"
    )
    print(
        f"Saved             : "
        f"{output_path}"
    )
    print()

    coverage = any_detected / total

    if coverage >= 0.90:
        print("EXCELLENT — excellent hand coverage.")
    elif coverage >= 0.75:
        print("GOOD — suitable for 3D reconstruction.")
    elif coverage >= 0.50:
        print("PARTIAL — interpolation can recover gaps.")
    else:
        print("LOW — source still needs better tracking.")

    print("=" * 70)


if __name__ == "__main__":
    extract()
