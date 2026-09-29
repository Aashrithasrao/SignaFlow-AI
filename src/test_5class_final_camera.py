from pathlib import Path
from collections import Counter, deque
import time

import cv2
import numpy as np
import tensorflow as tf

from src.camera_landmarks import CameraLandmarkExtractor
from src.preprocess_landmarks import resize_sequence, normalize_sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sign_model_5class_v2.keras"
)

CLASSES = np.array([
    "hello",
    "mom",
    "dad",
    "water",
    "food",
])

SEQUENCE_LENGTH = 32
FEATURES = 225

CONFIDENCE_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.10

HISTORY_SIZE = 5
STABLE_COUNT = 3

COMMIT_COOLDOWN = 1.0


def prepare_sequence(sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    sequence = resize_sequence(
        sequence,
        SEQUENCE_LENGTH
    )

    sequence = normalize_sequence(
        sequence
    )

    return sequence.astype(np.float32)


def main():

    print("=" * 70)
    print("SIGNAFLOW AI — FINAL 5 CLASS CAMERA TEST")
    print("=" * 70)

    print("\nModel:")
    print(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("5-class model loaded.")

    print("\nClasses:")
    for i, name in enumerate(CLASSES):
        print(f"  {i}: {name}")

    extractor = CameraLandmarkExtractor()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera."
        )

    sequence = []

    prediction_history = deque(
        maxlen=HISTORY_SIZE
    )

    state = "IDLE"
    displayed_word = ""

    current_prediction = "NONE"
    current_confidence = 0.0
    current_margin = 0.0

    last_commit_time = 0.0

    print("\nCamera started.")
    print("------------------------------------------")
    print("Q = Quit")
    print("R = Reset")
    print("------------------------------------------")
    print()
    print("IDLE is application state.")
    print("There is NO idle class in the model.")
    print()

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(
            frame,
            1
        )

        landmarks, result = extractor.extract(
            frame
        )

        sequence.append(
            landmarks
        )

        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        # ====================================================
        # PREDICTION
        # ====================================================

        if len(sequence) == SEQUENCE_LENGTH:

            prepared = prepare_sequence(
                sequence
            )

            model_input = np.expand_dims(
                prepared,
                axis=0
            )

            probabilities = model.predict(
                model_input,
                verbose=0
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

            current_prediction = prediction.upper()
            current_confidence = confidence
            current_margin = margin

            # =================================================
            # FILTER WEAK PREDICTIONS
            # =================================================

            if (
                confidence >= CONFIDENCE_THRESHOLD
                and margin >= MARGIN_THRESHOLD
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

                # =============================================
                # STABLE SIGN
                # =============================================

                if (
                    count >= STABLE_COUNT
                    and (
                        time.time()
                        - last_commit_time
                    ) >= COMMIT_COOLDOWN
                ):

                    if (
                        state != "COMMITTED"
                        or word.upper()
                        != displayed_word
                    ):

                        displayed_word = word.upper()

                        state = "COMMITTED"

                        last_commit_time = time.time()

                        print(
                            f"\nSIGN: "
                            f"{displayed_word} "
                            f"| confidence="
                            f"{confidence:.2f}"
                        )

                        prediction_history.clear()

            # =================================================
            # NO STABLE SIGN
            # =================================================

            else:

                if state != "COMMITTED":

                    state = "IDLE"
                    displayed_word = ""

                prediction_history.clear()

        # ====================================================
        # RETURN TO IDLE
        # ====================================================

        if (
            state == "COMMITTED"
            and (
                time.time()
                - last_commit_time
            ) > COMMIT_COOLDOWN
        ):

            if len(prediction_history) == 0:

                state = "IDLE"
                displayed_word = ""

        # ====================================================
        # DISPLAY
        # ====================================================

        if state == "IDLE":

            state_text = "IDLE"

        else:

            state_text = f"SIGN: {displayed_word}"

        cv2.rectangle(
            frame,
            (0, 0),
            (
                frame.shape[1],
                160
            ),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"STATE: {state_text}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Prediction: {current_prediction}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {current_confidence:.2f}",
            (20, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Margin: {current_margin:.2f}",
            (20, 123),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q: Quit   R: Reset",
            (
                20,
                frame.shape[0] - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "SignaFlow AI - 5 Class Final Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):

            sequence.clear()
            prediction_history.clear()

            state = "IDLE"
            displayed_word = ""

            current_prediction = "NONE"
            current_confidence = 0.0
            current_margin = 0.0

            last_commit_time = 0.0

            print("\nRESET -> IDLE")

    cap.release()

    extractor.landmarker.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
