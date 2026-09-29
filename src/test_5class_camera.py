from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

from src.camera_landmarks import CameraLandmarkExtractor
from src.preprocess_landmarks import resize_sequence, normalize_sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "sign_model_10class_v1.keras"

CLASSES = np.array([
    "hello",
    "mom",
    "dad",
    "water",
    "food",
])

SEQUENCE_LENGTH = 32

CONFIDENCE_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.12

PREDICTION_WINDOW = 6
STABLE_COUNT = 4

# Motion threshold used to decide whether the hands/body
# are actually moving enough to be considered a sign.
MOTION_THRESHOLD = 0.008

# Number of consecutive low-motion frames needed to release
# the current sign.
RELEASE_FRAMES = 12


def prepare_sequence(sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    sequence = resize_sequence(
        sequence,
        SEQUENCE_LENGTH,
    )

    sequence = normalize_sequence(
        sequence,
    )

    return sequence.astype(
        np.float32
    )


def main():

    print("=" * 70)
    print("SIGNAFLOW AI — 5 CLASS CAMERA TEST")
    print("=" * 70)

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("\nModel loaded:")
    print(MODEL_PATH)

    print("\nClasses:")
    for i, name in enumerate(CLASSES):
        print(f"  {i}: {name}")

    print("\nStarting MediaPipe Holistic...")

    extractor = CameraLandmarkExtractor()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera."
        )

    sequence = []

    prediction_history = deque(
        maxlen=PREDICTION_WINDOW
    )

    state = "IDLE"

    committed_word = ""

    current_prediction = "---"
    current_confidence = 0.0
    current_margin = 0.0
    current_motion = 0.0

    release_counter = 0

    previous_landmarks = None

    print("\nCamera started.")
    print("------------------------------------------")
    print("Q = Quit")
    print("R = Reset")
    print("------------------------------------------")

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(
            frame,
            1,
        )

        # ====================================================
        # LANDMARK EXTRACTION
        # ====================================================

        landmarks, result = extractor.extract(
            frame
        )

        # ====================================================
        # MOTION
        # ====================================================

        if previous_landmarks is not None:

            current_motion = float(
                np.mean(
                    np.abs(
                        landmarks
                        - previous_landmarks
                    )
                )
            )

        previous_landmarks = landmarks.copy()

        sequence.append(
            landmarks
        )

        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        # ====================================================
        # HAND PRESENCE
        # ====================================================

        left_hand_present = bool(
            result.left_hand_landmarks
        )

        right_hand_present = bool(
            result.right_hand_landmarks
        )

        hand_present = (
            left_hand_present
            or right_hand_present
        )

        # ====================================================
        # PREDICTION
        # ====================================================

        if len(sequence) == SEQUENCE_LENGTH:

            prepared = prepare_sequence(
                sequence
            )

            model_input = np.expand_dims(
                prepared,
                axis=0,
            )

            probabilities = model.predict(
                model_input,
                verbose=0,
            )[0]

            sorted_indices = np.argsort(
                probabilities
            )[::-1]

            top_index = int(
                sorted_indices[0]
            )

            second_index = int(
                sorted_indices[1]
            )

            confidence = float(
                probabilities[top_index]
            )

            second_confidence = float(
                probabilities[second_index]
            )

            margin = (
                confidence
                - second_confidence
            )

            prediction = CLASSES[
                top_index
            ]

            current_prediction = prediction
            current_confidence = confidence
            current_margin = margin

            # =================================================
            # IDLE STATE
            # =================================================

            if state == "IDLE":

                prediction_history.clear()

                # Need an actual hand and enough movement
                # before starting a sign.
                if (
                    hand_present
                    and current_motion
                    >= MOTION_THRESHOLD
                    and confidence
                    >= CONFIDENCE_THRESHOLD
                    and margin
                    >= MARGIN_THRESHOLD
                ):

                    prediction_history.append(
                        prediction
                    )

                    counts = Counter(
                        prediction_history
                    )

                    word, count = (
                        counts.most_common(1)[0]
                    )

                    if count >= STABLE_COUNT:

                        committed_word = (
                            word.upper()
                        )

                        state = "COMMITTED"

                        release_counter = 0

                        print(
                            f"\nCOMMITTED: "
                            f"{committed_word} "
                            f"(confidence={confidence:.2f})"
                        )

            # =================================================
            # COMMITTED STATE
            # =================================================

            elif state == "COMMITTED":

                # Do NOT add another word while the user
                # is holding/signing.
                #
                # Wait until the hand disappears OR motion
                # becomes very low.

                if (
                    not hand_present
                    or current_motion
                    < MOTION_THRESHOLD
                ):

                    release_counter += 1

                else:

                    release_counter = 0

                if release_counter >= RELEASE_FRAMES:

                    print(
                        f"RELEASED: "
                        f"{committed_word}"
                    )

                    committed_word = ""

                    prediction_history.clear()

                    state = "IDLE"

                    release_counter = 0

        # ====================================================
        # DISPLAY STATE
        # ====================================================

        if state == "IDLE":

            display_state = "IDLE"

        else:

            display_state = (
                f"SIGN: {committed_word}"
            )

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 155),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            f"STATE: {display_state}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Prediction: {current_prediction.upper()}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Confidence: {current_confidence:.2f}",
            (20, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Motion: {current_motion:.3f}",
            (20, 123),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Q: Quit   R: Reset",
            (20, 148),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "SignaFlow AI - 5 Class Test",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):

            sequence.clear()
            prediction_history.clear()

            state = "IDLE"
            committed_word = ""

            current_prediction = "---"
            current_confidence = 0.0
            current_margin = 0.0
            current_motion = 0.0

            release_counter = 0
            previous_landmarks = None

            print("\nRESET → IDLE")

    cap.release()

    extractor.landmarker.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
