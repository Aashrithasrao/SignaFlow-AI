import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "dataset/processed"
MODEL_DIR = "models"

X_PATH = f"{DATA_DIR}/X.npy"
Y_PATH = f"{DATA_DIR}/y.npy"
CLASSES_PATH = f"{DATA_DIR}/classes.npy"

MODEL_PATH = f"{MODEL_DIR}/sign_model_50_v2.keras"

SEQUENCE_LENGTH = 32
FEATURES = 225
NUM_CLASSES = 50

TEST_SIZE = 0.20
RANDOM_STATE = 42

EPOCHS = 80
BATCH_SIZE = 32


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SIGN LANGUAGE MODEL V2")
print("=" * 70)

print("\nLoading dataset...")

X = np.load(X_PATH)
y = np.load(Y_PATH)
classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)

print(f"X: {X.shape}")
print(f"y: {y.shape}")
print(f"Classes: {len(classes)}")


# ============================================================
# VALIDATION
# ============================================================

assert X.shape[1:] == (32, 225)
assert len(X) == len(y)
assert len(classes) == 50

if np.isnan(X).any():
    raise ValueError("NaN detected.")

if np.isinf(X).any():
    raise ValueError("Infinite values detected.")


# ============================================================
# STRATIFIED SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# LANDMARK AUGMENTATION
# ============================================================

def augment_landmarks(x):

    x = tf.cast(x, tf.float32)

    # Small coordinate noise
    noise = tf.random.normal(
        tf.shape(x),
        mean=0.0,
        stddev=0.008
    )

    x = x + noise

    # Small global scale variation
    scale = tf.random.uniform(
        [],
        minval=0.97,
        maxval=1.03
    )

    x = x * scale

    return x


# ============================================================
# TF DATASET
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)


def train_preprocess(x, y):

    x = augment_landmarks(x)

    return x, y


train_ds = (
    train_ds
    .shuffle(
        len(X_train),
        seed=RANDOM_STATE
    )
    .map(
        train_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


val_ds = (
    val_ds
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

unique_classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_train,
)

class_weights = {
    int(c): float(w)
    for c, w in zip(
        unique_classes,
        weights
    )
}


# ============================================================
# MODEL
# ============================================================

print("\nBuilding V2 model...")

model = Sequential([
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

    BatchNormalization(),

    Dropout(0.30),

    Bidirectional(
        LSTM(
            48,
            return_sequences=False
        )
    ),

    Dropout(0.35),

    Dense(
        96,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(0.30),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0007
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

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
        patience=5,
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
print("TRAINING V2")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)


# ============================================================
# EVALUATE
# ============================================================

print("\n" + "=" * 70)
print("V2 EVALUATION")
print("=" * 70)

loss, accuracy = model.evaluate(
    val_ds,
    verbose=0
)

print(
    f"\nValidation loss: {loss:.4f}"
)

print(
    f"Validation accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Validation accuracy (%): "
    f"{accuracy * 100:.2f}%"
)

print(
    "\nBest V2 model:"
)

print(
    MODEL_PATH
)

print("\n" + "=" * 70)
print("V2 TRAINING COMPLETE")
print("=" * 70)
