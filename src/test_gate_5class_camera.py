from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

from src.camera_landmarks import CameraLandmarkExtractor
from src.preprocess_landmarks import resize_sequence, normalize_sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GATE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sign_activity_gate_v1.keras"
)

SIGN_MODEL_PATH = (
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

# Activity gate
SIGN_THRESHOLD = 0.65
RELEASE_THRESHOLD = 0.40

# Sign classifier
SIGN_CONFIDENCE_THRESHOLD = 0.55
SIGN_MARGIN_THRESHOLD = 0.10

# Temporal confirmation
PREDICTION_WINDOW = 5
STABLE_COUNT = 3

# Release confirmation
RELEASE_COUNT_REQUIRED = 3


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
    print("SIGNAFLOW AI — ACTIVITY GATE + 5 CLASS")
    print("=" * 70)

    print("\nLoading activity gate:")
    print(GATE_MODEL_PATH)

    gate_model = tf.keras.models.load_model(
        GATE_MODEL_PATH
    )

    print("Activity gate loaded.")

    print("\nLoading sign model:")
    print(SIGN_MODEL_PATH)

    sign_model = tf.keras.models.load_model(
        SIGN_MODEL_PATH
    )

    print("5-class sign model loaded.")

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

    current_word = "NONE"

    gate_probability = 0.0

    sign_confidence = 0.0

    sign_margin = 0.0

    release_counter = 0

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
        # WAIT FOR 32 FRAMES
        # ====================================================

        if len(sequence) == SEQUENCE_LENGTH:

            prepared = prepare_sequence(
                sequence
            )

            model_input = np.expand_dims(
                prepared,
                axis=0
            )

            # =================================================
            # ACTIVITY GATE
            # =================================================

            gate_probability = float(
                gate_model.predict(
                    model_input,
                    verbose=0
                )[0][0]
            )

            # gate_probability:
            # 0 = NO_SIGN
            # 1 = SIGN

            # =================================================
            # NO SIGN / RELEASE
            # =================================================

            if gate_probability < RELEASE_THRESHOLD:

                if state == "COMMITTED":

                    release_counter += 1

                    if (
                        release_counter
                        >= RELEASE_COUNT_REQUIRED
                    ):

                        print(
                            f"RELEASED: "
                            f"{committed_word}"
                        )

                        state = "IDLE"

                        committed_word = ""

                        current_word = "NONE"

                        prediction_history.clear()

                        release_counter = 0

                else:

                    state = "IDLE"

                    committed_word = ""

                    current_word = "NONE"

                    prediction_history.clear()

                    release_counter = 0

            # =================================================
            # SIGN DETECTED
            # =================================================

            elif gate_probability >= SIGN_THRESHOLD:

                release_counter = 0

                # ------------------------------------------------
                # If already committed, DO NOT produce another
                # word until the user releases the sign.
                # ------------------------------------------------

                if state == "COMMITTED":

                    current_word = committed_word

                # ------------------------------------------------
                # New sign
                # ------------------------------------------------

                else:

                    probabilities = (
                        sign_model.predict(
                            model_input,
                            verbose=0
                        )[0]
                    )

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
                        probabilities[
                            top_index
                        ]
                    )

                    second_confidence = float(
                        probabilities[
                            second_index
                        ]
                    )

                    margin = (
                        confidence
                        - second_confidence
                    )

                    prediction = CLASSES[
                        top_index
                    ]

                    current_word = (
                        prediction.upper()
                    )

                    sign_confidence = confidence

                    sign_margin = margin

                    # --------------------------------------------
                    # Accept only a confident prediction
                    # --------------------------------------------

                    if (
                        confidence
                        >= SIGN_CONFIDENCE_THRESHOLD
                        and margin
                        >= SIGN_MARGIN_THRESHOLD
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

                        # ----------------------------------------
                        # TEMPORAL CONFIRMATION
                        # ----------------------------------------

                        if count >= STABLE_COUNT:

                            committed_word = (
                                word.upper()
                            )

                            state = "COMMITTED"

                            print(
                                f"\nCOMMITTED: "
                                f"{committed_word} "
                                f"| confidence="
                                f"{confidence:.2f} "
                                f"| gate="
                                f"{gate_probability:.2f}"
                            )

                            prediction_history.clear()

                    else:

                        prediction_history.clear()

            # =================================================
            # BETWEEN THRESHOLDS
            # =================================================

            else:

                # Keep the current state.
                # This creates a small hysteresis zone
                # between SIGN and NO_SIGN.

                pass

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
            (
                frame.shape[1],
                175
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
            f"Gate SIGN: {gate_probability:.2f}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Word: {current_word}",
            (20, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {sign_confidence:.2f}",
            (20, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Margin: {sign_margin:.2f}",
            (20, 151),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
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
            "SignaFlow AI - Activity Gate + 5 Class",
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

            current_word = "NONE"

            gate_probability = 0.0

            sign_confidence = 0.0

            sign_margin = 0.0

            release_counter = 0

            print("\nRESET -> IDLE")

    cap.release()

    extractor.landmarker.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
