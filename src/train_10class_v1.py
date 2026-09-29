import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

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

LANDMARK_DIR = (
    BASE_DIR
    / "dataset"
    / "train_landmarks_50"
)

OLD_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "sign_model_5class_v3.keras"
)

NEW_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "sign_model_10class_v1.keras"
)


# ============================================================
# 10 CLASSES
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

OLD_CLASSES = [
    "hello",
    "mom",
    "dad",
    "water",
    "food",
]


# ============================================================
# TRAINING CONFIG
# ============================================================

SEQUENCE_LENGTH = 32
FEATURES = 225

BATCH_SIZE = 32
EPOCHS = 60

RANDOM_STATE = 42


# ============================================================
# START
# ============================================================

print("=" * 70)

print(
    "SIGNAFLOW AI - 10 CLASS MODEL"
)

print(
    "Starting from existing 5-class V3 model"
)

print("=" * 70)


print(
    "\nTensorFlow:",
    tf.__version__
)

print(
    "Devices:",
    tf.config.list_physical_devices()
)


# ============================================================
# LOAD CSV
# ============================================================

print(
    "\nLoading CSV..."
)

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
# FILTER 10 CLASSES
# ============================================================

df = df[
    df["sign"].isin(CLASSES)
].copy()


print(
    "\n10-class distribution:"
)

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


print(
    "\nClass mapping:"
)

for name, index in class_to_index.items():

    print(
        f"{index} -> {name}"
    )


# ============================================================
# LOAD LANDMARK DATA
# ============================================================

print(
    "\nLoading landmark data..."
)

print(
    "This may take a while..."
)

X = []
y = []

missing = 0
failed = 0


for counter, row in enumerate(
    df.itertuples(index=False),
    start=1
):

    # CSV columns:
    #
    # path
    # participant_id
    # sequence_id
    # sign

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
    # FALLBACK TO LANDMARK DIRECTORY
    # --------------------------------------------------------

    if not landmark_path.exists():

        fallback = (
            LANDMARK_DIR
            / landmark_path.name
        )

        if fallback.exists():

            landmark_path = fallback


    # --------------------------------------------------------
    # MISSING FILE
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
        # by the successful V3 model.

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


# ============================================================
# DATA SUMMARY
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

print(
    "=" * 70
)


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

print(
    "\nLoaded class counts:"
)

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


print(
    "\nClass weights:"
)

for index, name in enumerate(CLASSES):

    print(
        f"{name:>8}: "
        f"{class_weights[index]:.4f}"
    )


# ============================================================
# LOAD EXISTING V3 MODEL
# ============================================================

print(
    "\nLoading existing V3 model..."
)

old_model = tf.keras.models.load_model(
    OLD_MODEL_PATH
)


print(
    "Loaded:",
    OLD_MODEL_PATH
)


print(
    "\nOriginal V3 model:"
)

old_model.summary()


# ============================================================
# CREATE FEATURE EXTRACTOR
# ============================================================

print(
    "\nCreating 10-class model..."
)


# We copy the complete V3 model
# except its final Dense(5) layer.

feature_model = tf.keras.Sequential(
    name="sign_model_10class"
)


for layer in old_model.layers[:-1]:

    feature_model.add(
        layer
    )


# ============================================================
# ADD NEW 10-CLASS OUTPUT
# ============================================================

new_output = tf.keras.layers.Dense(
    10,
    activation="softmax",
    name="classification_10",
)


feature_model.add(
    new_output
)


# ============================================================
# IMPORTANT:
# BUILD THE MODEL BEFORE get_weights()
# ============================================================

feature_model.build(
    (
        None,
        SEQUENCE_LENGTH,
        FEATURES,
    )
)


# ============================================================
# TRANSFER EXISTING 5 CLASS OUTPUT
# ============================================================

print(
    "\nTransferring existing 5-class output weights..."
)


old_output = old_model.layers[-1]


old_weights = (
    old_output.get_weights()
)


old_kernel = old_weights[0]

old_bias = old_weights[1]


# New Dense(10) is now built,
# so get_weights() works.

new_weights = (
    new_output.get_weights()
)


new_kernel = new_weights[0]

new_bias = new_weights[1]


# ------------------------------------------------------------
# COPY OLD 5 CLASS WEIGHTS
# ------------------------------------------------------------

new_kernel[:, :5] = (
    old_kernel
)

new_bias[:5] = (
    old_bias
)


# ------------------------------------------------------------
# Keep random initialization
# for the new 5 classes.
# ------------------------------------------------------------

new_output.set_weights(
    [
        new_kernel,
        new_bias,
    ]
)


print(
    "\nTransferred existing classes:"
)

for index, name in enumerate(
    OLD_CLASSES
):

    print(
        f"  {index}: {name}"
    )


print(
    "\nNew classes:"
)

for index, name in enumerate(
    CLASSES[5:],
    start=5,
):

    print(
        f"  {index}: {name}"
    )


# ============================================================
# MAKE ALL LAYERS TRAINABLE
# ============================================================

for layer in feature_model.layers:

    layer.trainable = True


# ============================================================
# COMPILE
# ============================================================

print(
    "\nCompiling model..."
)


model = feature_model


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ],
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print(
    "\n10-class model:"
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        NEW_MODEL_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1,

    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=10,

        restore_best_weights=True,

        verbose=1,

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=4,

        min_lr=1e-7,

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
    "STARTING 10-CLASS TRANSFER TRAINING"
)

print(
    "=" * 70
)


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


if NEW_MODEL_PATH.exists():

    model = tf.keras.models.load_model(
        NEW_MODEL_PATH
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print(
    "\nEvaluating validation set..."
)


val_loss, val_accuracy = model.evaluate(

    X_val,

    y_val,

    verbose=0,

)


print(
    "\n"
    + "=" * 70
)

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
    f"{NEW_MODEL_PATH}"
)

print(
    "=" * 70
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion matrix..."
)


predictions = model.predict(
    X_val,
    verbose=0
)


predicted_classes = np.argmax(
    predictions,
    axis=1
)


cm = confusion_matrix(

    y_val,

    predicted_classes,

    labels=np.arange(
        len(CLASSES)
    ),

)


print(
    "\n             "
    + " ".join(
        f"{name:>8}"
        for name in CLASSES
    )
)


for index, name in enumerate(
    CLASSES
):

    print(

        f"{name:>8}   "

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


for index, name in enumerate(
    CLASSES
):

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

        f"{name:>8}: "

        f"{correct}/{total} "

        f"({recall:.2f}%)"

    )


print(
    "\n10-class training complete."
)
