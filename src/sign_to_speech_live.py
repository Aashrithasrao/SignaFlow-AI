import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.camera_landmarks import CameraLandmarkExtractor
from src.preprocess_landmarks import normalize_sequence


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HYBRID_MODEL_PATH = PROJECT_ROOT / "models/sign_model_50_hybrid_cnn.keras"
V3_MODEL_PATH = PROJECT_ROOT / "models/sign_model_50_v3.keras"
MODEL_PATH = HYBRID_MODEL_PATH if HYBRID_MODEL_PATH.exists() else V3_MODEL_PATH
CLASSES_PATH = PROJECT_ROOT / "dataset/processed/classes.npy"

RUNTIME_DIR = PROJECT_ROOT / "runtime"
RESULT_PATH = RUNTIME_DIR / "sign_result.json"

SEQUENCE_LENGTH = 32
FEATURES = 225

CONFIDENCE_THRESHOLD = 0.45
STABILITY_WINDOW = 6

# Time required without a confident sign before auto-stop
IDLE_TIMEOUT = 5.0

# Same-sign duplicates are blocked by last_committed_sign.
# This cooldown is only used to ignore very short transition
# predictions immediately after a word is committed.
TRANSITION_COOLDOWN = 0.60

# Stronger stability required when changing from one word to another.
# The first word still uses the original 6/6 rule.
TRANSITION_WINDOW = 12
TRANSITION_MIN_COUNT = 10

CAMERA_INDEX = 0


# ============================================================
# HELPERS
# ============================================================

def ensure_runtime_dir():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def write_result(
    sentence,
    status,
    current_sign="",
    confidence=0.0,
):
    ensure_runtime_dir()

    data = {
        "sentence": sentence,
        "status": status,
        "current_sign": current_sign,
        "confidence": float(confidence),
        "timestamp": time.time(),
    }

    temp_path = RESULT_PATH.with_suffix(".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        temp_path.replace(RESULT_PATH)

    except Exception:
        pass


def load_classes():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Classes file not found:\n{CLASSES_PATH}"
        )

    classes = np.load(
        CLASSES_PATH,
        allow_pickle=True,
    )

    return [str(x) for x in classes]


def normalize_for_model(sequence):
    """
    Apply the SAME sequence normalization used during training.

    Input:
        (32, 225)

    Output:
        (32, 225)
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURES,
    ):
        raise ValueError(
            f"Unexpected sequence shape: "
            f"{sequence.shape}. "
            f"Expected "
            f"({SEQUENCE_LENGTH}, {FEATURES})."
        )

    try:
        normalized = normalize_sequence(
            sequence
        )
    except TypeError:
        # In case the preprocessing function expects
        # a different invocation style.
        normalized = sequence

    normalized = np.asarray(
        normalized,
        dtype=np.float32,
    )

    if normalized.shape != (
        SEQUENCE_LENGTH,
        FEATURES,
    ):
        normalized = normalized.reshape(
            SEQUENCE_LENGTH,
            FEATURES,
        )

    return normalized


def get_prediction(model, sequence):
    """
    Run V3 model prediction.
    """

    model_input = normalize_for_model(
        sequence
    )

    model_input = np.expand_dims(
        model_input,
        axis=0,
    )

    probabilities = model.predict(
        model_input,
        verbose=0,
    )[0]

    class_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[class_index]
    )

    return class_index, confidence


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SIGNAFLOW AI — LIVE SIGN TO SPEECH")
print("=" * 70)

print()
print("Loading V3 model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

classes = load_classes()

print(
    f"Model loaded: {MODEL_PATH.name}"
)

print(
    f"Classes loaded: {len(classes)}"
)

print(
    f"Input: ({SEQUENCE_LENGTH}, {FEATURES})"
)

print()


# ============================================================
# CAMERA + MEDIAPIPE
# ============================================================

extractor = CameraLandmarkExtractor()

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    extractor.landmarker.close()
    raise SystemExit(1)

print("Camera opened successfully.")
print()
print("Recognition started.")
print()
print("Make a sign and hold it naturally.")
print()
print("Press Q to stop manually.")
print()


# ============================================================
# STATE
# ============================================================

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)

prediction_history = deque(
    maxlen=STABILITY_WINDOW
)

# Used only when changing from one already-committed word to a
# different word. This filters brief MAN/MOM-style transition flips.
transition_history = deque(
    maxlen=TRANSITION_WINDOW
)

recognized_words = []

last_committed_sign = None
last_commit_time = 0.0

last_confident_time = time.time()

current_prediction = ""
current_confidence = 0.0

status = "RUNNING"

write_result(
    sentence="",
    status="RUNNING",
)


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # CAMERA FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print(
                "Camera frame could not be read."
            )
            status = "ERROR"
            break

        # ----------------------------------------------------
        # MIRROR CAMERA
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1,
        )

        # ----------------------------------------------------
        # EXTRACT LANDMARKS
        # ----------------------------------------------------

        # IMPORTANT:
        # extract() returns:
        #
        #     features, result
        #
        # NOT just features.

        landmarks, result = (
            extractor.extract(frame)
        )

        # ----------------------------------------------------
        # VALIDATE LANDMARK VECTOR
        # ----------------------------------------------------

        if landmarks is None:
            landmarks = np.zeros(
                FEATURES,
                dtype=np.float32,
            )

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32,
        )

        if landmarks.shape != (
            FEATURES,
        ):
            landmarks = landmarks.reshape(-1)

            if landmarks.size < FEATURES:

                padded = np.zeros(
                    FEATURES,
                    dtype=np.float32,
                )

                padded[
                    :landmarks.size
                ] = landmarks

                landmarks = padded

            elif landmarks.size > FEATURES:

                landmarks = landmarks[
                    :FEATURES
                ]

        # ----------------------------------------------------
        # ADD FRAME TO SEQUENCE
        # ----------------------------------------------------

        sequence_buffer.append(
            landmarks
        )

        # Temporary motion measurement.
        # This tells us how much the landmarks actually move
        # between frames, which we will use to distinguish
        # standing still from an intentional sign.
        if len(sequence_buffer) >= 2:
            previous_landmarks = np.asarray(
                sequence_buffer[-2],
                dtype=np.float32,
            )

            current_landmarks = np.asarray(
                sequence_buffer[-1],
                dtype=np.float32,
            )

            motion_value = float(
                np.mean(
                    np.abs(
                        current_landmarks
                        - previous_landmarks
                    )
                )
            )
        else:
            motion_value = 0.0

        # ----------------------------------------------------
        # TEMPORARY MOTION DIAGNOSTIC
        # ----------------------------------------------------
        if len(sequence_buffer) % 30 == 0:
            print(
                f"Motion value: {motion_value:.6f}"
            )

        # ----------------------------------------------------
        # DRAW MEDIAPIPE LANDMARKS
        # ----------------------------------------------------

        try:
            extractor.draw(
                frame,
                result,
            )
        except Exception:
            pass

        # ----------------------------------------------------
        # DEFAULT DISPLAY
        # ----------------------------------------------------

        display_sign = (
            current_prediction
            if current_confidence
            >= CONFIDENCE_THRESHOLD
            else "Waiting..."
        )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        if (
            len(sequence_buffer)
            == SEQUENCE_LENGTH
        ):

            sequence = np.asarray(
                sequence_buffer,
                dtype=np.float32,
            )

            try:

                class_index, confidence = (
                    get_prediction(
                        model,
                        sequence,
                    )
                )

                if (
                    class_index
                    < len(classes)
                ):

                    prediction = (
                        classes[
                            class_index
                        ]
                    )

                else:

                    prediction = ""

                current_prediction = (
                    prediction
                )

                current_confidence = (
                    confidence
                )

            except Exception as e:

                print(
                    "\nPrediction error:",
                    e,
                )

                current_prediction = ""
                current_confidence = 0.0

        # ----------------------------------------------------
        # CONFIDENT SIGN
        # ----------------------------------------------------

        if (
            current_prediction
            and current_confidence
            >= CONFIDENCE_THRESHOLD
        ):

            # Any confident prediction means
            # the user is actively signing.
            #
            # Therefore a sign being held for 8 or
            # 10 seconds will NOT trigger idle timeout.

            last_confident_time = (
                time.time()
            )

            now = time.time()

            # ------------------------------------------------
            # FIRST SIGN / NORMAL STABILITY
            # ------------------------------------------------

            # If there is no previously committed word, keep the
            # original 6-of-6 behavior. This makes the first sign
            # easy to recognize and avoids over-filtering HELLO,
            # MOM, etc.

            if last_committed_sign is None:

                prediction_history.append(
                    current_prediction
                )

                if (
                    len(prediction_history)
                    == STABILITY_WINDOW
                ):

                    counts = {}

                    for word in prediction_history:
                        counts[word] = (
                            counts.get(word, 0)
                            + 1
                        )

                    stable_sign = max(
                        counts,
                        key=counts.get,
                    )

                    stable_count = counts[
                        stable_sign
                    ]

                    if (
                        stable_count
                        == STABILITY_WINDOW
                    ):

                        recognized_words.append(
                            stable_sign
                        )

                        last_committed_sign = (
                            stable_sign
                        )

                        last_commit_time = now

                        sentence = " ".join(
                            recognized_words
                        )

                        print(
                            f"\n✓ COMMITTED: "
                            f"{stable_sign.upper()}"
                        )

                        print(
                            f"Sentence: "
                            f"{sentence}"
                        )

                        write_result(
                            sentence=sentence,
                            status="RUNNING",
                            current_sign=stable_sign,
                            confidence=current_confidence,
                        )

                        prediction_history.clear()
                        transition_history.clear()

            # ------------------------------------------------
            # NEXT SIGN / TRANSITION FILTER
            # ------------------------------------------------

            else:

                # Never commit the same word again while it is
                # being held. Only collect a candidate that differs
                # from the last committed word.
                if (
                    current_prediction
                    != last_committed_sign
                    and (
                        now - last_commit_time
                        >= TRANSITION_COOLDOWN
                    )
                ):

                    transition_history.append(
                        current_prediction
                    )

                    if (
                        len(transition_history)
                        == TRANSITION_WINDOW
                    ):

                        transition_counts = {}

                        for word in transition_history:
                            transition_counts[word] = (
                                transition_counts.get(word, 0)
                                + 1
                            )

                        candidate_sign = max(
                            transition_counts,
                            key=transition_counts.get,
                        )

                        candidate_count = (
                            transition_counts[
                                candidate_sign
                            ]
                        )

                        # A different sign must dominate 10 of
                        # the last 12 predictions. A short transition
                        # such as MAN -> MOM should therefore not
                        # become an extra sentence word.

                        if (
                            candidate_sign
                            != last_committed_sign
                            and candidate_count
                            >= TRANSITION_MIN_COUNT
                        ):

                            recognized_words.append(
                                candidate_sign
                            )

                            last_committed_sign = (
                                candidate_sign
                            )

                            last_commit_time = now

                            sentence = " ".join(
                                recognized_words
                            )

                            print(
                                f"\n✓ COMMITTED: "
                                f"{candidate_sign.upper()}"
                            )

                            print(
                                f"Sentence: "
                                f"{sentence}"
                            )

                            write_result(
                                sentence=sentence,
                                status="RUNNING",
                                current_sign=candidate_sign,
                                confidence=current_confidence,
                            )

                            transition_history.clear()
                            prediction_history.clear()

                else:
                    # Still holding the previous sign or still inside
                    # the short transition cooldown.
                    transition_history.clear()

        else:

            # ------------------------------------------------
            # NO CONFIDENT SIGN
            # ------------------------------------------------

            prediction_history.clear()

            current_prediction = ""

            current_confidence = 0.0

            # A genuine low-confidence/release period makes the
            # next occurrence of the same sign eligible again.
            #
            # We do NOT immediately reset on one weak frame because
            # that would make noisy tracking create duplicate words.

            if (
                last_committed_sign is not None
                and (
                    time.time()
                    - last_confident_time
                    >= 0.45
                )
            ):
                last_committed_sign = None
                transition_history.clear()
                prediction_history.clear()

        # ----------------------------------------------------
        # AUTO STOP AFTER 5 SEC OF NO CONFIDENT SIGN
        # ----------------------------------------------------

        idle_time = (
            time.time()
            - last_confident_time
        )

        if idle_time >= IDLE_TIMEOUT:

            print()
            print(
                "No confident sign detected "
                f"for {IDLE_TIMEOUT:.0f} seconds."
            )

            status = "AUTO_STOPPED"

            break

        # ----------------------------------------------------
        # UI TEXT
        # ----------------------------------------------------

        sentence = " ".join(
            recognized_words
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
            "SIGNAFLOW AI",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Sign: {display_sign.upper()}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Confidence: "
            f"{current_confidence * 100:.1f}%",
            (20, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Idle: {idle_time:.1f}s / "
            f"{IDLE_TIMEOUT:.0f}s",
            (20, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # SENTENCE
        # ----------------------------------------------------

        sentence_display = (
            sentence
            if sentence
            else "Start signing..."
        )

        cv2.putText(
            frame,
            f"Sentence: {sentence_display}",
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # SHOW CAMERA
        # ----------------------------------------------------

        cv2.imshow(
            "SignaFlow AI - Live Recognition",
            frame,
        )

        # ----------------------------------------------------
        # MANUAL STOP
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print()
            print(
                "Manual stop requested."
            )

            status = "STOPPED"

            break

        # ----------------------------------------------------
        # UPDATE RUNTIME RESULT
        # ----------------------------------------------------

        write_result(
            sentence=sentence,
            status="RUNNING",
            current_sign=current_prediction,
            confidence=current_confidence,
        )


finally:

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    try:
        extractor.landmarker.close()
    except Exception:
        pass

    final_sentence = " ".join(
        recognized_words
    )

    write_result(
        sentence=final_sentence,
        status=status,
        current_sign="",
        confidence=0.0,
    )

    print()
    print("=" * 70)
    print("RECOGNITION FINISHED")
    print("=" * 70)

    print()
    print(
        f"Status: {status}"
    )

    print()

    if final_sentence:

        print(
            "Recognized sentence:"
        )

        print(
            final_sentence
        )

    else:

        print(
            "No signs were recognized."
        )

    print()
