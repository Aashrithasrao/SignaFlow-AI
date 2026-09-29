import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from preprocess_landmarks import (
    load_landmarks,
    resize_sequence,
    normalize_sequence,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = (
    BASE_DIR
    / "dataset"
    / "train_50_classes.csv"
)

LANDMARK_DIR = (
    BASE_DIR
    / "dataset"
    / "train_landmarks_50"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "sign_model_10class_v2.keras"
)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
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


# ============================================================
# TRAINING CONFIG
# ============================================================

SEQUENCE_LENGTH = 32
FEATURES = 225

BATCH_SIZE = 32
EPOCHS = 100

RANDOM_STATE = 42


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SIGNAFLOW AI - 10 CLASS V2")
print("FRESH TRAINING - ALL 10 CLASSES TOGETHER")
print("=" * 70)

print("\nTensorFlow:", tf.__version__)

print(
    "Devices:",
    tf.config.list_physical_devices()
)


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading CSV...")

df = pd.read_csv(
    CSV_PATH
)

print(
    "CSV columns:",
    list(df.columns)
)

print(
    "Total CSV rows:",
    len(df)
)


# ============================================================
# FILTER CLASSES
# ============================================================

df = df[
    df["sign"].isin(CLASSES)
].copy()

df = df.reset_index(
    drop=True
)


print("\n10-class distribution:")

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
    name: index
    for index, name in enumerate(CLASSES)
}


print("\nClass mapping:")

for name, index in class_to_index.items():

    print(
        f"{index} -> {name}"
    )


# ============================================================
# LOAD LANDMARK DATA
# ============================================================

print("\nLoading landmark data...")
print("This may take a while...")

X = []
y = []

missing = 0
failed = 0


for counter, row in enumerate(
    df.itertuples(index=False),
    start=1
):

    # --------------------------------------------------------
    # CSV PATH
    # --------------------------------------------------------

    landmark_path = Path(
        str(row.path)
    )


    # --------------------------------------------------------
    # RESOLVE RELATIVE PATH
    # --------------------------------------------------------

    if not landmark_path.is_absolute():

        landmark_path = (
            BASE_DIR
            / landmark_path
        )


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not landmark_path.exists():

        fallback = (
            LANDMARK_DIR
            / landmark_path.name
        )

        if fallback.exists():

            landmark_path = fallback


    # --------------------------------------------------------
    # MISSING
    # --------------------------------------------------------

    if not landmark_path.exists():

        missing += 1
        continue


    # --------------------------------------------------------
    # LOAD + PREPROCESS
    # --------------------------------------------------------

    try:

        sequence = load_landmarks(
            landmark_path
        )


        # EXACT preprocessing used
        # by successful V3 model.

        sequence = resize_sequence(
            sequence,
            SEQUENCE_LENGTH
        )


        sequence = normalize_sequence(
            sequence
        )


        # ----------------------------------------------------
        # VERIFY SHAPE
        # ----------------------------------------------------

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURES
        ):

            failed += 1
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

        failed += 1

        print(
            f"Failed: "
            f"{landmark_path.name} "
            f"-> {error}"
        )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if counter % 250 == 0:

        print(
            f"Processed "
            f"{counter}/{len(df)}"
        )


# ============================================================
# NUMPY
# ============================================================

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int32
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "Loaded samples:",
    len(X)
)

print(
    "Missing files:",
    missing
)

print(
    "Failed files:",
    failed
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
        "No landmark data loaded."
    )


if X.shape[1:] != (
    SEQUENCE_LENGTH,
    FEATURES
):

    raise RuntimeError(
        f"Unexpected X shape: {X.shape}"
    )


# ============================================================
# CLASS COUNTS
# ============================================================

print("\nLoaded class counts:")

for index, name in enumerate(CLASSES):

    count = int(
        np.sum(
            y == index
        )
    )

    print(
        f"{name:>8}: {count}"
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print(
    "\nCreating stratified split..."
)

X_train, X_val, y_train, y_val = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
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

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(
        len(CLASSES)
    ),
    y=y_train,
)


class_weights = {
    index: float(weight)
    for index, weight
    in enumerate(weights)
}


print("\nClass weights:")

for index, name in enumerate(CLASSES):

    print(
        f"{name:>8}: "
        f"{class_weights[index]:.4f}"
    )


# ============================================================
# MODEL
# ============================================================

print(
    "\nBuilding fresh 10-class model..."
)


model = tf.keras.Sequential(
    [

        tf.keras.layers.Input(
            shape=(
                SEQUENCE_LENGTH,
                FEATURES
            )
        ),


        # ----------------------------------------------------
        # FIRST BIDIRECTIONAL LSTM
        # ----------------------------------------------------

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                64,
                return_sequences=True
            )
        ),


        tf.keras.layers.Dropout(
            0.30
        ),


        # ----------------------------------------------------
        # SECOND BIDIRECTIONAL LSTM
        # ----------------------------------------------------

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                32
            )
        ),


        tf.keras.layers.Dropout(
            0.30
        ),


        # ----------------------------------------------------
        # DENSE
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),


        tf.keras.layers.Dropout(
            0.30
        ),


        # ----------------------------------------------------
        # 10 CLASS OUTPUT
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            len(CLASSES),
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
    ]
)


print(
    "\nModel summary:"
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=15,

        restore_best_weights=True,

        mode="max",

        verbose=1,
    ),


    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=5,

        min_lr=1e-6,

        verbose=1,
    ),


    tf.keras.callbacks.ModelCheckpoint(

        filepath=MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1,
    ),
]


# ============================================================
# TRAIN
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "STARTING 10-CLASS TRAINING"
)

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

    callbacks=callbacks,

    shuffle=True,

    verbose=1,
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best saved model..."
)

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print(
    "\nEvaluating best model..."
)

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)


print(
    "\n"
    + "=" * 70
)

print(
    "FINAL VALIDATION RESULTS"
)

print("=" * 70)

print(
    f"Validation loss: "
    f"{loss:.4f}"
)

print(
    f"Validation accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

predictions = model.predict(
    X_val,
    verbose=0
)

y_pred = np.argmax(
    predictions,
    axis=1
)


cm = confusion_matrix(
    y_val,
    y_pred,
    labels=np.arange(
        len(CLASSES)
    )
)


print(
    "\nConfusion matrix:"
)

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification report:"
)

print(
    classification_report(
        y_val,
        y_pred,
        labels=np.arange(
            len(CLASSES)
        ),
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


print(
    "\n"
    + "=" * 70
)

print(
    "TRAINING COMPLETE"
)

print(
    "Model saved to:"
)

print(
    MODEL_PATH
)

print("=" * 70)
