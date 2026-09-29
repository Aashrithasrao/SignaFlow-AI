import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Allow imports when running:
# python src/train_5class_v3.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "sign_model_5class_v3.keras"

CLASSES = [
    "hello",
    "mom",
    "dad",
    "water",
    "food",
]

NUM_CLASSES = len(CLASSES)

SEQUENCE_LENGTH = 32
FEATURES = 225

RANDOM_STATE = 42

BATCH_SIZE = 32
EPOCHS = 80


# ============================================================
# GPU / DEVICE INFORMATION
# ============================================================

print("=" * 70)
print("SIGNaflow AI - 5 CLASS MODEL V3")
print("=" * 70)

print("\nTensorFlow:", tf.__version__)

devices = tf.config.list_physical_devices()
print("Devices:", devices)


# ============================================================
# LOAD DATAFRAME
# ============================================================

print("\nLoading CSV...")

df = pd.read_csv(CSV_PATH)

print("CSV columns:", list(df.columns))
print("Total CSV rows:", len(df))


# Keep ONLY our five classes.
df = df[
    df["sign"].isin(CLASSES)
].copy()

print("\n5-class samples:")
print(
    df["sign"]
    .value_counts()
    .reindex(CLASSES)
    .fillna(0)
)


# ============================================================
# CLASS MAPPING
# ============================================================

class_to_index = {
    class_name: index
    for index, class_name in enumerate(CLASSES)
}

print("\nClass mapping:")

for class_name, index in class_to_index.items():
    print(
        f"{index} -> {class_name}"
    )


# ============================================================
# LOAD LANDMARK DATA
# ============================================================

print("\nLoading landmark data...")
print("This can take a while because all 5 classes are being loaded.\n")

X = []
y = []

missing_files = 0
failed_files = 0

for counter, row in enumerate(
    df.itertuples(index=False),
    start=1
):

    # CSV structure:
    # path, participant_id, sequence_id, sign

    landmark_path = Path(
        str(row.path)
    )

    # Relative paths are relative to project root.
    if not landmark_path.is_absolute():
        landmark_path = BASE_DIR / landmark_path

    # Fallback:
    # use the filename inside train_landmarks_50.
    if not landmark_path.exists():

        fallback = (
            LANDMARK_DIR /
            landmark_path.name
        )

        if fallback.exists():
            landmark_path = fallback

    if not landmark_path.exists():

        missing_files += 1

        continue

    try:

        # ----------------------------------------------------
        # EXACT SAME PREPROCESSING AS THE SUCCESSFUL V2 MODEL
        # ----------------------------------------------------

        sequence = load_landmarks(
            landmark_path
        )

        sequence = resize_sequence(
            sequence,
            SEQUENCE_LENGTH
        )

        sequence = normalize_sequence(
            sequence
        )

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURES
        ):

            failed_files += 1

            print(
                f"Invalid shape: "
                f"{landmark_path.name} "
                f"{sequence.shape}"
            )

            continue

        X.append(
            sequence.astype(
                np.float32
            )
        )

        y.append(
            class_to_index[
                row.sign
            ]
        )

    except Exception as error:

        failed_files += 1

        print(
            f"Failed: "
            f"{landmark_path.name} "
            f"-> {error}"
        )

    if counter % 250 == 0:

        print(
            f"Processed "
            f"{counter}/{len(df)}"
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int32
)

print("\n" + "=" * 70)

print(
    "Loaded samples:",
    len(X)
)

print(
    "Missing files:",
    missing_files
)

print(
    "Failed files:",
    failed_files
)

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print("=" * 70)


if len(X) == 0:

    raise RuntimeError(
        "No training samples were loaded."
    )


if X.shape[1:] != (
    SEQUENCE_LENGTH,
    FEATURES
):

    raise RuntimeError(
        f"Unexpected X shape: "
        f"{X.shape}"
    )


# ============================================================
# SHOW ACTUAL CLASS COUNTS
# ============================================================

print("\nLoaded class distribution:")

for index, class_name in enumerate(CLASSES):

    count = int(
        np.sum(y == index)
    )

    print(
        f"{class_name:>8}: "
        f"{count}"
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print(
    "\nCreating stratified train/validation split..."
)

X_train, X_val, y_train, y_val = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y,
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=np.arange(
        NUM_CLASSES
    ),

    y=y_train,
)

class_weights = {
    index: float(weight)
    for index, weight
    in enumerate(
        class_weights_array
    )
}

print("\nClass weights:")

for index, class_name in enumerate(CLASSES):

    print(
        f"{class_name:>8}: "
        f"{class_weights[index]:.4f}"
    )


# ============================================================
# BUILD FRESH MODEL
# ============================================================

print(
    "\nBuilding NEW 5-class model..."
)

model = tf.keras.Sequential(

    [

        tf.keras.layers.Input(
            shape=(
                SEQUENCE_LENGTH,
                FEATURES
            )
        ),

        tf.keras.layers.Bidirectional(

            tf.keras.layers.LSTM(
                64,
                return_sequences=True
            )

        ),

        tf.keras.layers.Dropout(
            0.30
        ),

        tf.keras.layers.Bidirectional(

            tf.keras.layers.LSTM(
                32,
                return_sequences=False
            )

        ),

        tf.keras.layers.Dropout(
            0.30
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.30
        ),

        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        ),

    ]

)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ],

)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel summary:\n")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=MODEL_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1,

    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=15,

        restore_best_weights=True,

        verbose=1,

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=5,

        min_lr=1e-7,

        verbose=1,

    ),

]


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING FRESH 5-CLASS TRAINING")
print("=" * 70)

history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    shuffle=True,

    callbacks=callbacks,

    verbose=1,

)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best saved model..."
)

if MODEL_PATH.exists():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print(
    "\nEvaluating final validation set..."
)

val_loss, val_accuracy = model.evaluate(

    X_val,

    y_val,

    verbose=0,

)


print("\n" + "=" * 70)

print(
    f"Validation loss: "
    f"{val_loss:.4f}"
)

print(
    f"Validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Saved model: "
    f"{MODEL_PATH}"
)

print("=" * 70)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nGenerating confusion matrix..."
)

predictions = model.predict(
    X_val,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(

    y_val,

    predicted_classes,

    labels=np.arange(
        NUM_CLASSES
    ),

)

print(
    "\nConfusion matrix:"
)

print(
    "             "
    + " ".join(
        f"{name:>8}"
        for name in CLASSES
    )
)

for index, class_name in enumerate(CLASSES):

    print(

        f"{class_name:>8}   "

        + " ".join(

            f"{value:8d}"

            for value
            in cm[index]

        )

    )


# ============================================================
# PER-CLASS RECALL
# ============================================================

print(
    "\nPer-class results:"
)

for index, class_name in enumerate(CLASSES):

    total = np.sum(
        cm[index]
    )

    correct = cm[index][index]

    if total > 0:

        recall = (
            correct / total
        ) * 100

    else:

        recall = 0.0

    print(
        f"{class_name:>8}: "
        f"{correct}/{total} "
        f"({recall:.2f}%)"
    )


print(
    "\nTraining complete."
)
