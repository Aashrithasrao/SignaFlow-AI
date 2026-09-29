import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEQUENCE_LENGTH = 32

LANDMARK_TYPES = {
    "left_hand": 21,
    "right_hand": 21,
    "pose": 33,
}


# ---------------------------------------------------------
# Load one Parquet file
# ---------------------------------------------------------

def load_landmarks(file_path):
    df = pd.read_parquet(file_path)

    frames = sorted(df["frame"].unique())

    sequences = []

    for frame in frames:

        frame_data = df[df["frame"] == frame]

        features = []

        # Left hand
        for landmark_type, landmark_count in LANDMARK_TYPES.items():

            data = frame_data[
                frame_data["type"] == landmark_type
            ].sort_values("landmark_index")

            # Create empty landmark array
            landmarks = np.zeros(
                (landmark_count, 3),
                dtype=np.float32
            )

            # Fill available landmarks
            for _, row in data.iterrows():

                index = int(row["landmark_index"])

                if index >= landmark_count:
                    continue

                x = row["x"]
                y = row["y"]
                z = row["z"]

                # Replace missing values with 0
                if pd.isna(x):
                    x = 0.0

                if pd.isna(y):
                    y = 0.0

                if pd.isna(z):
                    z = 0.0

                landmarks[index] = [x, y, z]

            features.extend(landmarks.flatten())

        sequences.append(features)

    return np.asarray(sequences, dtype=np.float32)


# ---------------------------------------------------------
# Convert variable-length sequence → fixed length
# ---------------------------------------------------------

def resize_sequence(sequence, target_length=SEQUENCE_LENGTH):
    """
    Convert a variable-length landmark sequence
    to exactly target_length frames.

    Instead of zero-padding short sequences,
    uniformly resample the original frames.

    Examples:
        6 frames   -> 32 frames
        23 frames  -> 32 frames
        32 frames  -> 32 frames
        161 frames -> 32 frames
    """

    current_length = len(sequence)

    if current_length == 0:
        return np.zeros(
            (target_length, 225),
            dtype=np.float32
        )

    if current_length == target_length:
        return sequence.astype(np.float32)

    # Uniformly sample the complete original sequence.
    indices = np.linspace(
        0,
        current_length - 1,
        target_length
    )

    indices = np.round(indices).astype(int)

    return sequence[indices].astype(np.float32)


# ---------------------------------------------------------
# Normalize coordinates
# ---------------------------------------------------------

def normalize_sequence(sequence):

    sequence = sequence.copy()

    # For every frame, normalize x/y/z around the
    # coordinate center of the visible landmarks.

    for i in range(len(sequence)):

        frame = sequence[i]

        # Coordinates are arranged:
        # x, y, z, x, y, z, ...

        points = frame.reshape(-1, 3)

        # Ignore zero padding when calculating center
        valid = np.any(points != 0, axis=1)

        if not np.any(valid):
            continue

        center = points[valid].mean(axis=0)

        points[valid] -= center

        # Scale by maximum distance from center
        distances = np.linalg.norm(
            points[valid],
            axis=1
        )

        max_distance = distances.max()

        if max_distance > 0:
            points[valid] /= max_distance

        sequence[i] = points.flatten()

    return sequence


# ---------------------------------------------------------
# Main test
# ---------------------------------------------------------

if __name__ == "__main__":

    file_path = (
        "dataset/sample_landmarks/1000035562.parquet"
    )

    print("=" * 60)
    print("LANDMARK PREPROCESSING TEST")
    print("=" * 60)

    print("\nLoading:")
    print(file_path)

    sequence = load_landmarks(file_path)

    print("\nOriginal shape:")
    print(sequence.shape)

    sequence = resize_sequence(sequence)

    print("\nAfter resizing:")
    print(sequence.shape)

    sequence = normalize_sequence(sequence)

    print("\nAfter normalization:")
    print(sequence.shape)

    print("\nData type:")
    print(sequence.dtype)

    print("\nMinimum value:")
    print(sequence.min())

    print("Maximum value:")
    print(sequence.max())

    print("\nNaN values:")
    print(np.isnan(sequence).sum())

    print("\nInfinite values:")
    print(np.isinf(sequence).sum())

    print("\n" + "=" * 60)
    print("PREPROCESSING TEST COMPLETE")
    print("=" * 60)
