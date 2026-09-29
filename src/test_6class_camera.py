from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

from src.camera_landmarks import CameraLandmarkExtractor
from src.preprocess_landmarks import resize_sequence, normalize_sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "sign_model_6class_v2.keras"

CLASSES = np.array([
    "hello",
    "mom",
    "dad",
    "water",
    "food",
    "no_sign",
])

SEQUENCE_LENGTH = 32

CONFIDENCE_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.10

PREDICTION_WINDOW = 6
STABLE_COUNT = 4

NO_SIGN_THRESHOLD = 0.70


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

    return sequence.astype(
        np.float32
    )


def main():

    print("=" * 70)
    print("SIGNAFLOW AI — 6 CLASS CAMERA TEST")
    print("=" * 70)

    print("\nLoading model:")
    print(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded.")

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
        maxlen=PREDICTION_WINDOW
    )

    state = "IDLE"
    committed_word = ""

    current_prediction = "NO_SIGN"
    current_confidence = 0.0
    current_margin = 0.0

    release_counter = 0

    print("\nCamera started.")
    print("------------------------------------------")
    print("Q = Quit")
    print("R = Reset")
    print("------------------------------------------")
    print("\nExpected behavior:")
    print("NO SIGN  -> IDLE")
    print("SIGN     -> one word")
    print("RELEASE  -> IDLE")
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

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

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

            current_prediction = prediction
            current_confidence = confidence
            current_margin = margin

            # =================================================
            # NO_SIGN
            # =================================================

            if prediction == "no_sign":

                if state == "COMMITTED":

                    release_counter += 1

                    if release_counter >= 4:

                        print(
                            f"RELEASED: "
                            f"{committed_word}"
                        )

                        state = "IDLE"
                        committed_word = ""
                        prediction_history.clear()
                        release_counter = 0

                else:

                    state = "IDLE"
                    committed_word = ""
                    prediction_history.clear()
                    release_counter = 0

            # =================================================
            # POSSIBLE SIGN
            # =================================================

            else:

                # ------------------------------------------------
                # IDLE → candidate sign
                # ------------------------------------------------

                if state == "IDLE":

                    if (
                        confidence
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

                    else:

                        prediction_history.clear()

                # ------------------------------------------------
                # COMMITTED → wait for NO_SIGN
                # ------------------------------------------------

                elif state == "COMMITTED":

                    # Do not commit another sign.
                    # Stay locked until the model sees NO_SIGN.

                    release_counter = 0

        # =====================================================
        # DISPLAY
        # =====================================================

        if state == "IDLE":

            state_text = "IDLE"

        else:

            state_text = (
                f"SIGN: {committed_word}"
            )

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 145),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"STATE: {state_text}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Model: {current_prediction.upper()}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
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
            (20, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "SignaFlow AI - 6 Class Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):

            sequence.clear()
            prediction_history.clear()

            state = "IDLE"
            committed_word = ""

            current_prediction = "NO_SIGN"
            current_confidence = 0.0
            current_margin = 0.0

            release_counter = 0

            print("\nRESET -> IDLE")

    cap.release()

    extractor.landmarker.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
