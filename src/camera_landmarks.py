import cv2
import numpy as np
import mediapipe as mp

from pathlib import Path

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "models/mediapipe/holistic_landmarker.task"
)

FEATURES = 225

LEFT_HAND_POINTS = 21
RIGHT_HAND_POINTS = 21
POSE_POINTS = 33


# ============================================================
# CAMERA LANDMARK EXTRACTOR
# ============================================================

class CameraLandmarkExtractor:

    def __init__(self):

        print("=" * 70)
        print("SIGNAFLOW — MEDIAPIPE HOLISTIC")
        print("=" * 70)

        model_path = Path(MODEL_PATH)

        if not model_path.exists():

            raise FileNotFoundError(
                f"\nMediaPipe model not found:\n"
                f"{model_path}\n\n"
                f"Download the holistic .task model first."
            )

        # ----------------------------------------------------
        # Base options
        # ----------------------------------------------------

        base_options = BaseOptions(
            model_asset_path=str(
                model_path
            )
        )

        # ----------------------------------------------------
        # Holistic options
        # ----------------------------------------------------

        options = (
            vision.HolisticLandmarkerOptions(
                base_options=base_options,

                running_mode=(
                    VisionTaskRunningMode.IMAGE
                ),

                min_face_detection_confidence=0.5,
                min_pose_detection_confidence=0.5,
                min_hand_landmarks_confidence=0.5,

                output_face_blendshapes=False,
                output_segmentation_mask=False,
            )
        )

        # ----------------------------------------------------
        # Create landmarker
        # ----------------------------------------------------

        self.landmarker = (
            vision.HolisticLandmarker
            .create_from_options(
                options
            )
        )

        print(
            "Holistic Landmarker loaded."
        )

        print(
            f"Feature size: {FEATURES}"
        )


    # ========================================================
    # EXTRACT FEATURES
    # ========================================================

    def extract(self, frame):

        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # MediaPipe Image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb
        )

        # ----------------------------------------------------
        # Run Holistic
        # ----------------------------------------------------

        result = (
            self.landmarker.detect(
                mp_image
            )
        )

        # ----------------------------------------------------
        # Empty landmark arrays
        # ----------------------------------------------------

        left_hand = np.zeros(
            (LEFT_HAND_POINTS, 3),
            dtype=np.float32
        )

        right_hand = np.zeros(
            (RIGHT_HAND_POINTS, 3),
            dtype=np.float32
        )

        pose = np.zeros(
            (POSE_POINTS, 3),
            dtype=np.float32
        )

        # ====================================================
        # HANDS
        # ====================================================

        # MediaPipe Holistic returns hand landmarks
        # in separate left/right lists.

        if result.left_hand_landmarks:

            landmarks = (
                result.left_hand_landmarks
            )

            for i, landmark in enumerate(
                landmarks
            ):

                if i >= LEFT_HAND_POINTS:
                    break

                left_hand[i] = [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                ]

        if result.right_hand_landmarks:

            landmarks = (
                result.right_hand_landmarks
            )

            for i, landmark in enumerate(
                landmarks
            ):

                if i >= RIGHT_HAND_POINTS:
                    break

                right_hand[i] = [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                ]

        # ====================================================
        # POSE
        # ====================================================

        if result.pose_landmarks:

            landmarks = (
                result.pose_landmarks
            )

            for i, landmark in enumerate(
                landmarks
            ):

                if i >= POSE_POINTS:
                    break

                pose[i] = [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                ]

        # ====================================================
        # EXACT 225 FEATURE VECTOR
        # ====================================================

        features = np.concatenate(
            [
                left_hand.flatten(),
                right_hand.flatten(),
                pose.flatten(),
            ]
        )

        return features.astype(
            np.float32
        ), result


    # ========================================================
    # DRAW LANDMARKS
    # ========================================================

    def draw(
        self,
        frame,
        result
    ):

        height, width = (
            frame.shape[:2]
        )

        # ----------------------------------------------------
        # Draw pose
        # ----------------------------------------------------

        if result.pose_landmarks:

            for landmark in (
                result.pose_landmarks
            ):

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                if (
                    0 <= x < width
                    and 0 <= y < height
                ):

                    cv2.circle(
                        frame,
                        (x, y),
                        2,
                        (0, 255, 0),
                        -1
                    )

        # ----------------------------------------------------
        # Draw left hand
        # ----------------------------------------------------

        if result.left_hand_landmarks:

            for landmark in (
                result.left_hand_landmarks
            ):

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                if (
                    0 <= x < width
                    and 0 <= y < height
                ):

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (255, 0, 0),
                        -1
                    )

        # ----------------------------------------------------
        # Draw right hand
        # ----------------------------------------------------

        if result.right_hand_landmarks:

            for landmark in (
                result.right_hand_landmarks
            ):

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                if (
                    0 <= x < width
                    and 0 <= y < height
                ):

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 0, 255),
                        -1
                    )

        return frame


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.landmarker.close()


# ============================================================
# CAMERA TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nStarting camera test..."
    )

    extractor = (
        CameraLandmarkExtractor()
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        extractor.close()

        raise RuntimeError(
            "Could not open webcam."
        )

    print(
        "\nCamera opened."
    )

    print(
        "Move your hands in front of the camera."
    )

    print(
        "Press Q to quit."
    )

    frame_count = 0

    try:

        while True:

            success, frame = (
                camera.read()
            )

            if not success:

                print(
                    "Failed to read camera frame."
                )

                break

            # ------------------------------------------------
            # Extract landmarks
            # ------------------------------------------------

            features, result = (
                extractor.extract(
                    frame
                )
            )

            frame_count += 1

            # ------------------------------------------------
            # Draw
            # ------------------------------------------------

            frame = extractor.draw(
                frame,
                result
            )

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Features: {len(features)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Frames: {frame_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "SignaFlow - Live Landmark Test",
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                break

    finally:

        camera.release()

        cv2.destroyAllWindows()

        extractor.close()

        print(
            "\nCamera test finished."
        )
