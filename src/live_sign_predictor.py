import cv2
import numpy as np
import tensorflow as tf

from collections import deque, Counter
from pathlib import Path

from src.camera_landmarks import CameraLandmarkExtractor
from src.preprocess_landmarks import normalize_sequence


# ============================================================
# CONFIGURATION
# ============================================================

HYBRID_MODEL_PATH = Path("models/sign_model_50_hybrid_cnn.keras")
V3_MODEL_PATH = Path("models/sign_model_50_v3.keras")
MODEL_PATH = HYBRID_MODEL_PATH if HYBRID_MODEL_PATH.exists() else V3_MODEL_PATH

CLASSES_PATH = Path(
    "dataset/processed/classes.npy"
)

SEQUENCE_LENGTH = 32
FEATURES = 225

# ------------------------------------------------------------
# Prediction stability
# ------------------------------------------------------------

PREDICTION_HISTORY = 10

# A prediction must reach this confidence before
# it is allowed into the stability system.
CONFIDENCE_THRESHOLD = 0.45

# Number of matching predictions required.
MIN_STABLE_COUNT = 6

# Run inference every N frames.
PREDICTION_INTERVAL = 3

# Frames to wait after accepting a sign.
COOLDOWN_FRAMES = 20


# ============================================================
# LOAD CLASSES
# ============================================================

print("=" * 70)
print("SIGNAFLOW — STABLE LIVE SIGN PREDICTOR")
print("=" * 70)

print("\nLoading classes...")

classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)

print(
    f"Classes loaded: {len(classes)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    f"Model loaded: {MODEL_PATH}"
)

print(
    f"Model input shape: {model.input_shape}"
)

print(
    f"Model output shape: {model.output_shape}"
)


# ============================================================
# VALIDATE MODEL
# ============================================================

expected_shape = (
    SEQUENCE_LENGTH,
    FEATURES
)

if model.input_shape[-2:] != expected_shape:

    raise ValueError(
        "\nUnexpected model input shape.\n"
        f"Found:    {model.input_shape}\n"
        f"Expected: (?, {SEQUENCE_LENGTH}, {FEATURES})"
    )


if model.output_shape[-1] != len(classes):

    raise ValueError(
        "\nModel/class mismatch.\n"
        f"Model outputs: {model.output_shape[-1]}\n"
        f"Classes:       {len(classes)}"
    )


# ============================================================
# MEDIA PIPE
# ============================================================

extractor = CameraLandmarkExtractor()


# ============================================================
# ROLLING FRAME BUFFER
# ============================================================

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)


# ============================================================
# PREDICTION HISTORY
# ============================================================

prediction_history = deque(
    maxlen=PREDICTION_HISTORY
)


# ============================================================
# STATE
# ============================================================

displayed_sign = "Waiting..."

displayed_confidence = 0.0

candidate_sign = None

candidate_count = 0

last_accepted_sign = None

cooldown_remaining = 0

last_raw_prediction = "Waiting..."

last_raw_confidence = 0.0


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sequence(sequence):
    """
    Run one prediction on a 32-frame sequence.

    Returns:
        raw_prediction
        raw_confidence
        stable_prediction
        stable_confidence
        stable_count
        status
    """

    global candidate_sign
    global candidate_count
    global last_accepted_sign
    global cooldown_remaining
    global displayed_sign
    global displayed_confidence

    # --------------------------------------------------------
    # Buffer check
    # --------------------------------------------------------

    if len(sequence) < SEQUENCE_LENGTH:

        return (
            last_raw_prediction,
            last_raw_confidence,
            displayed_sign,
            displayed_confidence,
            candidate_count,
            "BUFFERING"
        )

    # --------------------------------------------------------
    # Convert sequence
    # --------------------------------------------------------

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != expected_shape:

        return (
            last_raw_prediction,
            last_raw_confidence,
            displayed_sign,
            displayed_confidence,
            candidate_count,
            "BAD SHAPE"
        )

    # --------------------------------------------------------
    # Training-compatible normalization
    # --------------------------------------------------------

    normalized = normalize_sequence(
        sequence
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    model_input = np.expand_dims(
        normalized,
        axis=0
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    probabilities = model.predict(
        model_input,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # Best prediction
    # --------------------------------------------------------

    class_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[class_index]
    )

    prediction = str(
        classes[class_index]
    )

    # --------------------------------------------------------
    # Save raw prediction
    # --------------------------------------------------------

    last_raw_prediction = prediction

    last_raw_confidence = confidence

    # --------------------------------------------------------
    # Cooldown
    # --------------------------------------------------------

    if cooldown_remaining > 0:

        cooldown_remaining -= 1

        return (
            prediction,
            confidence,
            displayed_sign,
            displayed_confidence,
            candidate_count,
            "COOLDOWN"
        )

    # --------------------------------------------------------
    # Low confidence
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        # Do NOT add unstable predictions
        # to the history.

        prediction_history.clear()

        candidate_sign = None

        candidate_count = 0

        return (
            prediction,
            confidence,
            displayed_sign,
            displayed_confidence,
            0,
            "UNCERTAIN"
        )

    # --------------------------------------------------------
    # Add high-confidence prediction
    # --------------------------------------------------------

    prediction_history.append(
        prediction
    )

    # --------------------------------------------------------
    # Majority voting
    # --------------------------------------------------------

    counts = Counter(
        prediction_history
    )

    most_common = (
        counts.most_common(1)
    )

    if not most_common:

        return (
            prediction,
            confidence,
            displayed_sign,
            displayed_confidence,
            0,
            "WAITING"
        )

    majority_sign, majority_count = (
        most_common[0]
    )

    # --------------------------------------------------------
    # Candidate stability
    # --------------------------------------------------------

    if majority_sign == candidate_sign:

        candidate_count = majority_count

    else:

        candidate_sign = majority_sign

        candidate_count = majority_count

    # --------------------------------------------------------
    # Stable sign
    # --------------------------------------------------------

    if candidate_count >= MIN_STABLE_COUNT:

        # Don't repeatedly accept the same sign.

        if (
            candidate_sign
            != last_accepted_sign
        ):

            displayed_sign = (
                candidate_sign
            )

            displayed_confidence = (
                confidence
            )

            last_accepted_sign = (
                candidate_sign
            )

            # Clear history after acceptance.
            #
            # This forces the next sign to establish
            # itself instead of inheriting the previous one.

            prediction_history.clear()

            candidate_sign = None

            candidate_count = 0

            cooldown_remaining = (
                COOLDOWN_FRAMES
            )

            return (
                prediction,
                confidence,
                displayed_sign,
                displayed_confidence,
                MIN_STABLE_COUNT,
                "STABLE"
            )

        else:

            return (
                prediction,
                confidence,
                displayed_sign,
                displayed_confidence,
                candidate_count,
                "HOLD"
            )

    # --------------------------------------------------------
    # Still stabilizing
    # --------------------------------------------------------

    return (
        prediction,
        confidence,
        displayed_sign,
        displayed_confidence,
        candidate_count,
        "STABILIZING"
    )


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    extractor.close()

    raise RuntimeError(
        "\nCould not open webcam.\n"
        "Check macOS camera permissions."
    )


# ============================================================
# CAMERA SETTINGS
# ============================================================

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# START
# ============================================================

print("\n" + "=" * 70)
print("LIVE CAMERA READY")
print("=" * 70)

print(
    "Hold one sign steadily for several seconds."
)

print(
    "The system requires repeated predictions "
    "before accepting a sign."
)

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
)

print(
    f"Stable votes required: "
    f"{MIN_STABLE_COUNT}/{PREDICTION_HISTORY}"
)

print(
    f"Prediction interval: "
    f"every {PREDICTION_INTERVAL} frames"
)

print(
    "Press Q to quit."
)

print("=" * 70)


# ============================================================
# MAIN LOOP
# ============================================================

frame_count = 0

raw_prediction = "Waiting..."

raw_confidence = 0.0

stable_prediction = "Waiting..."

stable_confidence = 0.0

stability_count = 0

status = "BUFFERING"


try:

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "Could not read camera frame."
            )

            break

        # ----------------------------------------------------
        # Mirror camera
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # Extract MediaPipe landmarks
        # ----------------------------------------------------

        features, result = (
            extractor.extract(
                frame
            )
        )

        frame_count += 1

        # ----------------------------------------------------
        # Validate feature vector
        # ----------------------------------------------------

        if len(features) != FEATURES:

            print(
                f"\nWARNING: expected "
                f"{FEATURES} features, "
                f"got {len(features)}"
            )

            continue

        # ----------------------------------------------------
        # Add frame
        # ----------------------------------------------------

        sequence_buffer.append(
            features
        )

        # ----------------------------------------------------
        # Draw landmarks
        # ----------------------------------------------------

        frame = extractor.draw(
            frame,
            result
        )

        # ----------------------------------------------------
        # Run prediction periodically
        # ----------------------------------------------------

        if (
            len(sequence_buffer)
            == SEQUENCE_LENGTH
            and frame_count
            % PREDICTION_INTERVAL
            == 0
        ):

            (
                raw_prediction,
                raw_confidence,
                stable_prediction,
                stable_confidence,
                stability_count,
                status
            ) = predict_sequence(
                sequence_buffer
            )

        # ====================================================
        # DISPLAY
        # ====================================================

        height, width = (
            frame.shape[:2]
        )

        # ----------------------------------------------------
        # Top information panel
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (15, 15),
            (530, 215),
            (15, 15, 15),
            -1
        )

        # ----------------------------------------------------
        # Current stable sign
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"SIGN: {stable_prediction.upper()}",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Stable confidence
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"CONFIDENCE: "
                f"{stable_confidence * 100:.1f}%"
            ),
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Raw prediction
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"RAW: "
                f"{raw_prediction.upper()} "
                f"{raw_confidence * 100:.1f}%"
            ),
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # Stability
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"STABILITY: "
                f"{stability_count}/"
                f"{MIN_STABLE_COUNT}"
            ),
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Buffer
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"BUFFER: "
                f"{len(sequence_buffer)}/"
                f"{SEQUENCE_LENGTH}"
            ),
            (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # AI status
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "AI: LIVE",
            (width - 150, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        cv2.putText(
            frame,
            status,
            (width - 180, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # Progress bar
        # ----------------------------------------------------

        bar_x = 30
        bar_y = height - 50
        bar_width = 400
        bar_height = 15

        progress = (
            len(sequence_buffer)
            / SEQUENCE_LENGTH
        )

        progress_width = int(
            bar_width * progress
        )

        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y
            ),
            (
                bar_x + bar_width,
                bar_y + bar_height
            ),
            (70, 70, 70),
            -1
        )

        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y
            ),
            (
                bar_x + progress_width,
                bar_y + bar_height
            ),
            (255, 255, 255),
            -1
        )

        # ----------------------------------------------------
        # Show
        # ----------------------------------------------------

        cv2.imshow(
            "SignaFlow AI - Stable Live Recognition",
            frame
        )

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

finally:

    camera.release()

    cv2.destroyAllWindows()

    extractor.close()

    print(
        "\n" + "=" * 70
    )

    print(
        "LIVE SIGN PREDICTOR STOPPED"
    )

    print(
        "=" * 70
    )
