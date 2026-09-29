import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
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

# V3 model is saved separately so V2 is not overwritten
MODEL_PATH = f"{MODEL_DIR}/sign_model_50_v3.keras"

SEQUENCE_LENGTH = 32
FEATURES = 225
NUM_CLASSES = 50

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

EPOCHS = 80
BATCH_SIZE = 32

LEARNING_RATE = 0.0005


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SIGNALFLOW AI - SIGN LANGUAGE MODEL V3 TRAINING")
print("=" * 70)


# ============================================================
# ENVIRONMENT
# ============================================================

print("\nTensorFlow version:", tf.__version__)

# Make sure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING PROCESSED DATASET")
print("=" * 70)

print("\nLoading X.npy...")
X = np.load(X_PATH)

print("Loading y.npy...")
y = np.load(Y_PATH)

print("Loading classes.npy...")
classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)


print("\nDataset:")
print(f"X shape       : {X.shape}")
print(f"y shape       : {y.shape}")
print(f"Classes       : {len(classes)}")
print(f"Samples       : {len(X):,}")


# ============================================================
# VALIDATE DATA
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING DATASET")
print("=" * 70)

if X.ndim != 3:
    raise ValueError(
        f"Expected X to have 3 dimensions, got {X.ndim}"
    )

if X.shape[1:] != (
    SEQUENCE_LENGTH,
    FEATURES
):
    raise ValueError(
        f"Expected X shape (*, {SEQUENCE_LENGTH}, {FEATURES}), "
        f"got {X.shape}"
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

if y.min() < 0 or y.max() >= NUM_CLASSES:
    raise ValueError(
        f"Invalid class labels found. "
        f"Expected labels from 0 to {NUM_CLASSES - 1}."
    )

print("\nDataset validation PASSED.")

print(f"Samples : {len(X):,}")
print(f"Frames  : {X.shape[1]}")
print(f"Features: {X.shape[2]}")
print(f"Classes : {len(classes)}")

print(f"NaN     : {np.isnan(X).sum()}")
print(f"Inf     : {np.isinf(X).sum()}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FULL DATASET CLASS DISTRIBUTION")
print("=" * 70)

for index, class_name in enumerate(classes):
    count = np.sum(y == index)

    print(
        f"{index:2d} "
        f"{str(class_name):15s} "
        f"{count:4d}"
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
    f"\nTraining samples   : {len(X_train):,}"
)

print(
    f"Validation samples : {len(X_val):,}"
)

print(
    f"Training percentage   : {len(X_train) / len(X) * 100:.1f}%"
)

print(
    f"Validation percentage : {len(X_val) / len(X) * 100:.1f}%"
)


# ============================================================
# TRAINING CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

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

print("\n" + "=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

class_ids = np.unique(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=y_train,
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        class_ids,
        class_weights_array
    )
}

print("\nClass weights:")

for class_id, weight in class_weights.items():

    print(
        f"{class_id:2d} "
        f"{str(classes[class_id]):15s} "
        f"{weight:.4f}"
    )


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING V3 MODEL")
print("=" * 70)

model = Sequential(
    [
        Input(
            shape=(
                SEQUENCE_LENGTH,
                FEATURES
            )
        ),

        Bidirectional(
            LSTM(
                96,
                return_sequences=True
            )
        ),

        Dropout(0.35),

        Bidirectional(
            LSTM(
                64,
                return_sequences=False
            )
        ),

        Dropout(0.35),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.35),

        Dense(
            NUM_CLASSES,
            activation="softmax"
        ),
    ]
)


# ============================================================
# COMPILE
# ============================================================

print("\n" + "=" * 70)
print("COMPILING MODEL")
print("=" * 70)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel architecture:")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURING CALLBACKS")
print("=" * 70)

callbacks = [

    EarlyStopping(
        monitor="val_accuracy",
        patience=12,
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


print("\nBest model will be saved to:")
print(MODEL_PATH)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING V3 TRAINING")
print("=" * 70)

print(f"\nEpochs       : {EPOCHS}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Train samples: {len(X_train):,}")
print(f"Val samples  : {len(X_val):,}")

print("\nTraining started...\n")


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
print("V3 MODEL EVALUATION")
print("=" * 70)

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0,
)


print(
    f"\nValidation loss       : {loss:.4f}"
)

print(
    f"Validation accuracy   : {accuracy:.4f}"
)

print(
    f"Validation accuracy % : {accuracy * 100:.2f}%"
)


# ============================================================
# TRAINING HISTORY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

best_epoch = np.argmax(
    history.history["val_accuracy"]
) + 1

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_val_loss = min(
    history.history["val_loss"]
)

train_accuracy = history.history["accuracy"][-1]

print(
    f"\nBest epoch             : {best_epoch}"
)

print(
    f"Best validation acc.   : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best validation loss   : "
    f"{best_val_loss:.4f}"
)

print(
    f"Final training acc.    : "
    f"{train_accuracy * 100:.2f}%"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(
    f"\nInput shape  : {model.input_shape}"
)

print(
    f"Output shape : {model.output_shape}"
)

print(
    f"Parameters   : {model.count_params():,}"
)

print(
    f"Model path   : {MODEL_PATH}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("V3 TRAINING COMPLETE")
print("=" * 70)

print("\nBest V3 model saved to:")
print(MODEL_PATH)

print("\nDataset used:")
print(f"Samples : {len(X):,}")
print(f"Classes : {len(classes)}")
print(f"Shape   : {X.shape}")

print("\nFinal validation accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nNext step:")
print("Test the V3 model with the live camera predictor.")

print("=" * 70)
