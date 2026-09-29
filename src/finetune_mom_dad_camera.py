import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from preprocess_landmarks import (
    load_landmarks,
    resize_sequence,
    normalize_sequence,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "dataset" / "train_50_classes.csv"
LANDMARK_DIR = BASE_DIR / "dataset" / "train_landmarks_50"

CAMERA_DIR = BASE_DIR / "dataset" / "camera_mom_dad"

MODEL_PATH = BASE_DIR / "models" / "sign_model_5class_v2.keras"
OUTPUT_MODEL = BASE_DIR / "models" / "sign_model_5class_camera_v1.keras"

CLASSES = ["hello", "mom", "dad", "water", "food"]

SEQUENCE_LENGTH = 32
FEATURES = 225

CAMERA_WEIGHT = 4.0
BATCH_SIZE = 32
EPOCHS = 30


# ============================================================
# LOAD ORIGINAL 5-CLASS DATA
# ============================================================

print("Loading original landmark data...")

df = pd.read_csv(CSV_PATH)

# Keep only the five classes used by the current model.
df = df[df["sign"].isin(CLASSES)].copy()

class_to_idx = {
    "hello": 0,
    "mom": 1,
    "dad": 2,
    "water": 3,
    "food": 4,
}

X_original = []
y_original = []

missing = 0

for row in df.itertuples(index=False):

    # CSV columns:
    # path, participant_id, sequence_id, sign
    landmark_path = Path(str(row.path))

    # If path is relative, make it relative to the project.
    if not landmark_path.is_absolute():
        landmark_path = BASE_DIR / landmark_path

    # Fallback to the landmark directory if necessary.
    if not landmark_path.exists():
        fallback = LANDMARK_DIR / landmark_path.name

        if fallback.exists():
            landmark_path = fallback

    if not landmark_path.exists():
        missing += 1
        continue

    try:
        sequence = load_landmarks(landmark_path)

        # EXACT SAME preprocessing used for the successful V2 model.
        sequence = resize_sequence(
            sequence,
            SEQUENCE_LENGTH
        )

        sequence = normalize_sequence(sequence)

        if sequence.shape != (SEQUENCE_LENGTH, FEATURES):
            print(
                f"Skipping invalid shape: "
                f"{landmark_path.name} -> {sequence.shape}"
            )
            continue

        X_original.append(
            sequence.astype(np.float32)
        )

        y_original.append(
            class_to_idx[row.sign]
        )

    except Exception as e:
        print(
            f"Skipping {landmark_path.name}: {e}"
        )


print(f"Original samples loaded: {len(X_original)}")
print(f"Missing original files: {missing}")

if not X_original:
    raise RuntimeError(
        "No original landmark samples were loaded."
    )

X_original = np.asarray(
    X_original,
    dtype=np.float32
)

y_original = np.asarray(
    y_original,
    dtype=np.int32
)


# ============================================================
# LOAD CAMERA MOM/DAD DATA
# ============================================================

print("\nLoading camera MOM/DAD samples...")

camera_X = []
camera_y = []

for class_name in ["mom", "dad"]:

    class_dir = CAMERA_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Camera directory not found: {class_dir}"
        )

    files = sorted(
        class_dir.glob("*.npy")
    )

    print(
        f"{class_name.upper()}: "
        f"{len(files)} samples"
    )

    for file_path in files:

        arr = np.load(
            file_path
        ).astype(np.float32)

        if arr.shape != (
            SEQUENCE_LENGTH,
            FEATURES
        ):
            print(
                f"Skipping invalid camera sample: "
                f"{file_path.name} -> {arr.shape}"
            )
            continue

        camera_X.append(arr)

        camera_y.append(
            class_to_idx[class_name]
        )


camera_X = np.asarray(
    camera_X,
    dtype=np.float32
)

camera_y = np.asarray(
    camera_y,
    dtype=np.int32
)

print(
    f"Camera samples loaded: "
    f"{len(camera_X)}"
)

if len(camera_X) == 0:
    raise RuntimeError(
        "No camera MOM/DAD samples were loaded."
    )


# ============================================================
# SPLIT CAMERA DATA
# 80% TRAIN
# 20% TEST
# ============================================================

camera_train_X, camera_test_X, \
camera_train_y, camera_test_y = train_test_split(
    camera_X,
    camera_y,
    test_size=0.20,
    random_state=42,
    stratify=camera_y,
)

print(
    f"Camera training samples: "
    f"{len(camera_train_X)}"
)

print(
    f"Camera test samples: "
    f"{len(camera_test_X)}"
)


# ============================================================
# COMBINE ORIGINAL + CAMERA TRAINING DATA
# ============================================================

X_train = np.concatenate(
    [
        X_original,
        camera_train_X
    ],
    axis=0
)

y_train = np.concatenate(
    [
        y_original,
        camera_train_y
    ],
    axis=0
)

# Original samples = weight 1
# Camera samples = weight 4
sample_weights = np.ones(
    len(X_train),
    dtype=np.float32
)

sample_weights[
    len(X_original):
] = CAMERA_WEIGHT

print(
    f"\nCombined training samples: "
    f"{len(X_train)}"
)

print(
    f"Camera sample weight: "
    f"{CAMERA_WEIGHT}"
)


# ============================================================
# LOAD EXISTING 91% MODEL
# ============================================================

print(
    "\nLoading existing 5-class model..."
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Existing model loaded."
)

model.summary()


# ============================================================
# FREEZE MODEL
# ============================================================

for layer in model.layers:
    layer.trainable = False


# Unfreeze the final classification section.
for layer in model.layers[-3:]:
    layer.trainable = True


print(
    "\nTrainable layers:"
)

for layer in model.layers:

    if layer.trainable:
        print(
            "  ",
            layer.name
        )


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        OUTPUT_MODEL,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]


# ============================================================
# TRAIN
# ============================================================

print(
    "\nStarting camera fine-tuning...\n"
)

history = model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    validation_data=(
        camera_test_X,
        camera_test_y
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=callbacks,
    verbose=1,
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

if OUTPUT_MODEL.exists():

    print(
        "\nLoading best saved model..."
    )

    model = tf.keras.models.load_model(
        OUTPUT_MODEL
    )


# ============================================================
# FINAL CAMERA TEST
# ============================================================

print(
    "\nEvaluating on held-out camera samples..."
)

camera_loss, camera_accuracy = model.evaluate(
    camera_test_X,
    camera_test_y,
    verbose=0,
)

print(
    f"\nCamera test loss: "
    f"{camera_loss:.4f}"
)

print(
    f"Camera test accuracy: "
    f"{camera_accuracy * 100:.2f}%"
)

print(
    f"Saved: "
    f"{OUTPUT_MODEL}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

predictions = model.predict(
    camera_test_X,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

cm = confusion_matrix(
    camera_test_y,
    predicted_classes,
    labels=[
        0,
        1,
        2,
        3,
        4
    ],
)

print(
    "\nCamera confusion matrix:"
)

print(
    "             "
    + " ".join(
        f"{c:>8}"
        for c in CLASSES
    )
)

for i, class_name in enumerate(CLASSES):

    print(
        f"{class_name:>8}   "
        + " ".join(
            f"{value:8d}"
            for value in cm[i]
        )
    )


print(
    "\nFine-tuning complete."
)
