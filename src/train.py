import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "dataset/processed"
MODEL_DIR = "models"

X_PATH = f"{DATA_DIR}/X.npy"
Y_PATH = f"{DATA_DIR}/y.npy"
CLASSES_PATH = f"{DATA_DIR}/classes.npy"

MODEL_PATH = f"{MODEL_DIR}/sign_model_50.keras"

SEQUENCE_LENGTH = 32
FEATURES = 225
NUM_CLASSES = 50

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

EPOCHS = 60
BATCH_SIZE = 32


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SIGN LANGUAGE MODEL TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed dataset...")

X = np.load(X_PATH)
y = np.load(Y_PATH)
classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)


print("\nDataset:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {len(classes)}")


# ============================================================
# VALIDATE DATA
# ============================================================

if X.ndim != 3:
    raise ValueError(
        f"Expected X to have 3 dimensions, got {X.ndim}"
    )


if X.shape[1:] != (
    SEQUENCE_LENGTH,
    FEATURES
):
    raise ValueError(
        f"Expected X shape (*, {SEQUENCE_LENGTH}, "
        f"{FEATURES}), got {X.shape}"
    )


if len(X) != len(y):
    raise ValueError(
        "X and y have different numbers of samples."
    )


if len(classes) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"found {len(classes)}."
    )


if np.isnan(X).any():
    raise ValueError(
        "Dataset contains NaN values."
    )


if np.isinf(X).any():
    raise ValueError(
        "Dataset contains infinite values."
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / VALIDATION SPLIT")
print("=" * 70)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


print(
    f"\nTraining samples: "
    f"{len(X_train):,}"
)

print(
    f"Validation samples: "
    f"{len(X_val):,}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")

for index, class_name in enumerate(classes):

    count = np.sum(
        y_train == index
    )

    print(
        f"{index:2d} "
        f"{str(class_name):15s} "
        f"{count:4d}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        np.unique(y_train),
        class_weights_array
    )
}

print("Class weights calculated.")


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)


model = Sequential(
    [

        Bidirectional(
            LSTM(
                64,
                return_sequences=True
            ),
            input_shape=(
                SEQUENCE_LENGTH,
                FEATURES
            ),
        ),

        Dropout(0.30),

        Bidirectional(
            LSTM(
                32,
                return_sequences=False
            )
        ),

        Dropout(0.30),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.30),

        Dense(
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
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
]


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
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

    verbose=1,
)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)


loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0,
)


print(
    f"\nValidation loss: "
    f"{loss:.4f}"
)

print(
    f"Validation accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Validation accuracy (%): "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# BEST MODEL
# ============================================================

print("\nBest model saved to:")

print(
    MODEL_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

