import sys
from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sign_model_10class_v2.keras"
)


# ============================================================
# IMPORTS
# ============================================================

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.camera_landmarks import CameraLandmarkExtractor
from src.preprocess_landmarks import (
    resize_sequence,
    normalize_sequence,
)


# ============================================================
# CLASSES
# ============================================================

CLASSES = np.array(
    [
        "hello",
        "mom",
        "dad",
        "water",
        "food",
        "yes",
        "no",
        "please",
        "go",
        "home",
    ]
)


# ============================================================
# CONFIG
# ============================================================

SEQUENCE_LENGTH = 32

CONFIDENCE_THRESHOLD = 0.45

PREDICTION_HISTORY = 10

MIN_STABLE_COUNT = 5

CAMERA_INDEX = 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SIGNAFLOW AI - 10 CLASS CAMERA TEST")
    print("=" * 60)

    print(
        "\nModel:",
        MODEL_PATH
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading 10-class model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    print(
        "Output classes:",
        model.output_shape[-1]
    )

    if model.output_shape[-1] != len(CLASSES):

        raise RuntimeError(
            f"Model has "
            f"{model.output_shape[-1]} outputs, "
            f"but tester has "
            f"{len(CLASSES)} classes."
        )


    # ========================================================
    # LANDMARK EXTRACTOR
    # ========================================================

    print(
        "\nLoading MediaPipe Holistic Landmarker..."
    )

    extractor = CameraLandmarkExtractor()

    print(
        "Landmark extractor ready."
    )


    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera."
        )


    # ========================================================
    # BUFFERS
    # ========================================================

    sequence_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    prediction_history = deque(
        maxlen=PREDICTION_HISTORY
    )


    current_prediction = "Waiting..."

    current_confidence = 0.0


    print(
        "\n"
        + "-" * 60
    )

    print(
        "CAMERA TEST STARTED"
    )

    print(
        "-" * 60
    )

    print(
        "\nShow one sign clearly to the camera."
    )

    print(
        "Available signs:"
    )

    for index, name in enumerate(CLASSES):

        print(
            f"  {index}: {name}"
        )

    print(
        "\nQ = Quit"
    )

    print(
        "R = Reset"
    )

    print(
        "-" * 60
    )


    # ========================================================
    # CAMERA LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "Could not read camera frame."
            )

            break


        # ----------------------------------------------------
        # MIRROR CAMERA
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # EXTRACT LANDMARKS
        # ----------------------------------------------------

        try:

            extracted = extractor.extract(
                frame
            )

            # Your CameraLandmarkExtractor
            # returns two values:
            #
            # landmarks, result
            #
            # We only need landmarks here.

            if isinstance(extracted, tuple):
                landmarks = extracted[0]
                mp_result = extracted[1]
            else:
                landmarks = extracted
                mp_result = None

        except Exception as error:

            cv2.putText(
                frame,
                "Landmark extraction error",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            cv2.imshow(
                "SignaFlow AI - 10 Class Camera Test",
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

            continue

        # ----------------------------------------------------
        # CONVERT LANDMARKS
        # ----------------------------------------------------

        try:
            landmarks = np.asarray(
                landmarks,
                dtype=np.float32
            )
        except Exception:
            landmarks = None

        # ----------------------------------------------------
        # VALID LANDMARK FRAME
        # ----------------------------------------------------

        if (
            landmarks is not None
            and landmarks.shape == (225,)
        ):
            sequence_buffer.append(
                landmarks
            )

        # ====================================================
        # PREDICTION
        # ====================================================

        if len(sequence_buffer) >= SEQUENCE_LENGTH:
            sequence = np.asarray(
                sequence_buffer,
                dtype=np.float32
            )

            sequence = resize_sequence(
                sequence,
                SEQUENCE_LENGTH
            )

            sequence = normalize_sequence(
                sequence
            )

            input_data = np.expand_dims(
                sequence.astype(
                    np.float32
                ),
                axis=0
            )

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            predictions = model.predict(
                input_data,
                verbose=0
            )[0]

            # Anatomical check for NO (duck-beak handshape):
            if mp_result is not None:
                active_hands = [h for h in [mp_result.right_hand_landmarks, mp_result.left_hand_landmarks] if h]
                is_duck_beak = False
                for hand_lms in active_hands:
                    try:
                        palm_scale = ((hand_lms[9].x - hand_lms[0].x)**2 + (hand_lms[9].y - hand_lms[0].y)**2)**0.5
                        if palm_scale > 1e-4:
                            # Finger extensions: distance from MCP (base knuckle) to fingertip / palm_scale
                            index_ext = ((hand_lms[8].x - hand_lms[5].x)**2 + (hand_lms[8].y - hand_lms[5].y)**2)**0.5 / palm_scale
                            mid_ext = ((hand_lms[12].x - hand_lms[9].x)**2 + (hand_lms[12].y - hand_lms[9].y)**2)**0.5 / palm_scale
                            pinky_ext = ((hand_lms[20].x - hand_lms[17].x)**2 + (hand_lms[20].y - hand_lms[17].y)**2)**0.5 / palm_scale

                            # Distances to thumb tip (4)
                            d_index_thumb = ((hand_lms[8].x - hand_lms[4].x)**2 + (hand_lms[8].y - hand_lms[4].y)**2)**0.5 / palm_scale
                            d_mid_thumb = ((hand_lms[12].x - hand_lms[4].x)**2 + (hand_lms[12].y - hand_lms[4].y)**2)**0.5 / palm_scale

                            # 1. Closed Fist (YES): All fingers curled
                            if index_ext < 0.68 and mid_ext < 0.68 and pinky_ext < 0.72:
                                detected_shape = "fist"
                                break

                            # 2. Pointing Forward / Outward (GO): Index extended, Middle curled
                            elif index_ext > 0.75 and mid_ext < 0.65 and pinky_ext < 0.70:
                                detected_shape = "pointing"
                                break

                            # 3. Duck Beak (NO): Index AND Middle extended together, tapping Thumb tip
                            elif index_ext > 0.65 and mid_ext > 0.65 and (d_index_thumb < 0.65 or d_mid_thumb < 0.65):
                                detected_shape = "duck_beak"
                                break
                    except Exception:
                        pass

                classes_list = list(CLASSES)
                yes_idx = classes_list.index("yes") if "yes" in classes_list else -1
                no_idx = classes_list.index("no") if "no" in classes_list else -1
                go_idx = classes_list.index("go") if "go" in classes_list else -1

                if detected_shape == "fist" and yes_idx >= 0:
                    predictions[yes_idx] = max(predictions[yes_idx] * 2.2, 0.45)
                    if no_idx >= 0: predictions[no_idx] *= 0.15
                    if go_idx >= 0: predictions[go_idx] *= 0.15
                    predictions = predictions / np.sum(predictions)

                elif detected_shape == "pointing" and go_idx >= 0:
                    predictions[go_idx] = max(predictions[go_idx] * 2.2, 0.45)
                    if no_idx >= 0: predictions[no_idx] *= 0.15
                    if yes_idx >= 0: predictions[yes_idx] *= 0.15
                    predictions = predictions / np.sum(predictions)

                elif detected_shape == "duck_beak" and no_idx >= 0:
                    predictions[no_idx] = max(predictions[no_idx] * 2.2, 0.45)
                    if yes_idx >= 0: predictions[yes_idx] *= 0.15
                    if go_idx >= 0: predictions[go_idx] *= 0.15
                    predictions = predictions / np.sum(predictions)

            top_index = int(
                np.argmax(
                    predictions
                )
            )

            confidence = float(
                predictions[
                    top_index
                ]
            )

            # ------------------------------------------------
            # CONFIDENCE & STABILITY FILTER
            # ------------------------------------------------
            eff_thresh = 0.30 if CLASSES[top_index] in ["no", "go", "yes"] else CONFIDENCE_THRESHOLD
            if confidence >= eff_thresh:
                prediction_history.append(
                    top_index
                )

                counts = Counter(
                    prediction_history
                )

                stable_index, stable_count = (
                    counts.most_common(1)[0]
                )

                # NO, GO, YES commit cleanly in 2 frames
                req_count = 2 if CLASSES[stable_index] in ["no", "go", "yes"] else 3
                if stable_count >= req_count:
                    current_prediction = (
                        CLASSES[
                            stable_index
                        ]
                    )

                    current_confidence = float(
                        predictions[
                            stable_index
                        ]
                    )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (630, 115),
            (0, 0, 0),
            -1,
        )


        cv2.putText(
            frame,
            "SignaFlow AI - 10 Class Test",
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )


        cv2.putText(
            frame,
            f"Prediction: {current_prediction}",
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )


        cv2.putText(
            frame,
            (
                f"Confidence: "
                f"{current_confidence * 100:.1f}%"
            ),
            (25, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


        # ----------------------------------------------------
        # FRAME COUNT
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"Frames: "
                f"{len(sequence_buffer)}/32"
            ),
            (
                20,
                frame.shape[0] - 45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


        cv2.putText(
            frame,
            "Q = Quit   R = Reset",
            (
                20,
                frame.shape[0] - 18
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "SignaFlow AI - 10 Class Camera Test",
            frame
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if key == ord("q"):

            break


        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        if key == ord("r"):

            sequence_buffer.clear()

            prediction_history.clear()

            current_prediction = (
                "Waiting..."
            )

            current_confidence = 0.0

            print(
                "Prediction buffer reset."
            )


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    print(
        "\nCamera test ended."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

