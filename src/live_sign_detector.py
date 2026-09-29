import numpy as np
import tensorflow as tf
from collections import deque, Counter
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/sign_model_50_v2.keras"
CLASSES_PATH = "dataset/processed/classes.npy"

SEQUENCE_LENGTH = 32
FEATURES = 225

CONFIDENCE_THRESHOLD = 0.55
STABLE_PREDICTIONS = 5


# ============================================================
# LIVE SIGN DETECTOR
# ============================================================

class LiveSignDetector:

    def __init__(self):

        print("=" * 60)
        print("LOADING SIGN LANGUAGE MODEL")
        print("=" * 60)

        self.model = tf.keras.models.load_model(
            MODEL_PATH
        )

        self.classes = np.load(
            CLASSES_PATH,
            allow_pickle=True
        )

        self.buffer = deque(
            maxlen=SEQUENCE_LENGTH
        )

        self.prediction_history = deque(
            maxlen=STABLE_PREDICTIONS
        )

        print(
            f"Model: {MODEL_PATH}"
        )

        print(
            f"Classes: {len(self.classes)}"
        )

        print(
            "Model loaded successfully."
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.buffer.clear()

        self.prediction_history.clear()


    # ========================================================
    # ADD FRAME
    # ========================================================

    def add_frame(self, landmarks):

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Validate frame
        # ----------------------------------------------------

        if landmarks.shape != (
            FEATURES,
        ):

            raise ValueError(
                f"Expected frame shape "
                f"({FEATURES},), "
                f"got {landmarks.shape}"
            )

        self.buffer.append(
            landmarks
        )

        # ----------------------------------------------------
        # Need 32 frames
        # ----------------------------------------------------

        if len(self.buffer) < SEQUENCE_LENGTH:

            return {
                "ready": False,
                "word": None,
                "confidence": 0.0,
            }

        # ----------------------------------------------------
        # Create sequence
        # ----------------------------------------------------

        sequence = np.asarray(
            self.buffer,
            dtype=np.float32
        )

        sequence = np.expand_dims(
            sequence,
            axis=0
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        probabilities = self.model.predict(
            sequence,
            verbose=0
        )[0]

        prediction_index = int(
            np.argmax(
                probabilities
            )
        )

        confidence = float(
            probabilities[
                prediction_index
            ]
        )

        word = str(
            self.classes[
                prediction_index
            ]
        )

        # ----------------------------------------------------
        # Confidence filtering
        # ----------------------------------------------------

        if confidence < CONFIDENCE_THRESHOLD:

            return {
                "ready": True,
                "word": None,
                "confidence": confidence,
                "raw_prediction": word,
            }

        # ----------------------------------------------------
        # Temporal smoothing
        # ----------------------------------------------------

        self.prediction_history.append(
            word
        )

        counts = Counter(
            self.prediction_history
        )

        stable_word, stable_count = (
            counts.most_common(1)[0]
        )

        # ----------------------------------------------------
        # Require repeated prediction
        # ----------------------------------------------------

        if stable_count < 3:

            return {
                "ready": True,
                "word": None,
                "confidence": confidence,
                "raw_prediction": word,
            }

        return {
            "ready": True,
            "word": stable_word,
            "confidence": confidence,
            "raw_prediction": word,
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nTesting LiveSignDetector..."
    )

    detector = LiveSignDetector()

    print(
        "\nDetector initialized."
    )

    print(
        "Waiting for camera frames..."
    )

    print(
        "\nLive detector test complete."
    )
