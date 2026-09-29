import os
import json
import time
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, precision_recall_fscore_support

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
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

MODEL_PATH = f"{MODEL_DIR}/sign_model_50_hybrid_cnn.keras"
METRICS_PATH = f"{MODEL_DIR}/hybrid_cnn_metrics.json"

SEQUENCE_LENGTH = 32
FEATURES = 225
NUM_CLASSES = 50

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0008


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SIGNAFLOW AI — HYBRID 1D-CNN + BiLSTM MODEL TRAINING")
print("=" * 70)

print(f"TensorFlow version: {tf.__version__}")
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed dataset...")
X = np.load(X_PATH)
y = np.load(Y_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {len(classes)}")


# ============================================================
# VALIDATE DATA
# ============================================================

if X.ndim != 3 or X.shape[1:] != (SEQUENCE_LENGTH, FEATURES):
    raise ValueError(f"Expected X shape (*, {SEQUENCE_LENGTH}, {FEATURES}), got {X.shape}")

if len(classes) != NUM_CLASSES:
    raise ValueError(f"Expected {NUM_CLASSES} classes, got {len(classes)}")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nTraining samples   : {len(X_train):,}")
print(f"Validation samples : {len(X_val):,}")


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_ids = np.unique(y_train)
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=y_train,
)
class_weights = {
    int(cid): float(w) for cid, w in zip(class_ids, class_weights_array)
}


# ============================================================
# HYBRID 1D-CNN + BiLSTM MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING HYBRID 1D-CNN + BiLSTM ARCHITECTURE")
print("=" * 70)

model = Sequential(
    [
        Input(shape=(SEQUENCE_LENGTH, FEATURES)),

        # Conv Block 1: Local temporal-spatial feature extraction
        Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        Dropout(0.20),

        # Conv Block 2: Higher-level temporal gesture features
        Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),  # (32 frames -> 16 pooled time steps)
        Dropout(0.25),

        # Recurrent Layer: Global bidirectional sequence context
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.35),

        # Dense Classification Head
        Dense(128, activation="relu"),
        Dropout(0.35),
        Dense(NUM_CLASSES, activation="softmax"),
    ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
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
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1,
    ),
]


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time / 60:.2f} minutes.")


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION ON VALIDATION SPLIT")
print("=" * 70)

# Load best saved weights
best_model = tf.keras.models.load_model(MODEL_PATH)
val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)

y_prob = best_model.predict(X_val, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

p_w, r_w, f1_w, _ = precision_recall_fscore_support(
    y_val, y_pred, average="weighted", zero_division=0
)
p_m, r_m, f1_m, _ = precision_recall_fscore_support(
    y_val, y_pred, average="macro", zero_division=0
)

# Confidence metrics for stability in sentences
confidences = np.max(y_prob, axis=1)
avg_confidence = float(np.mean(confidences))
high_confidence_rate = float(np.mean(confidences >= 0.50))

print(f"\nValidation Loss            : {val_loss:.4f}")
print(f"Validation Accuracy        : {val_acc * 100:.2f}%")
print(f"Weighted F1-Score          : {f1_w * 100:.2f}%")
print(f"Macro F1-Score             : {f1_m * 100:.2f}%")
print(f"Average Prediction Conf.   : {avg_confidence * 100:.2f}%")
print(f"Predictions with Conf >=50%: {high_confidence_rate * 100:.2f}%")

metrics_summary = {
    "model_name": "Hybrid 1D-CNN + BiLSTM",
    "val_loss": float(val_loss),
    "val_accuracy": float(val_acc),
    "weighted_precision": float(p_w),
    "weighted_recall": float(r_w),
    "weighted_f1": float(f1_w),
    "macro_f1": float(f1_m),
    "training_time_seconds": float(training_time),
    "avg_confidence": avg_confidence,
    "high_confidence_rate": high_confidence_rate,
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics_summary, f, indent=2)

print(f"\nSaved metrics summary to: {METRICS_PATH}")
