import os
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.preprocess_landmarks import (
    load_landmarks,
    resize_sequence,
    normalize_sequence,
)

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "dataset/train_5_classes.csv"
LANDMARK_DIR = Path("dataset/train_landmarks_50")

MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/sign_model_5class_v2.keras"

CLASSES = np.array([
    "hello",
    "mom",
    "dad",
    "water",
    "food",
])

SEQUENCE_LENGTH = 32
FEATURES = 225
NUM_CLASSES = 5

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 0.0005

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SIGNALFLOW AI - 5 CLASS MODEL V2")
print("TRAINING FROM SCRATCH")
print("=" * 70)

print("\nClasses:")
for i, name in enumerate(CLASSES):
    print(f"{i}: {name}")

print(f"\nEpochs        : {EPOCHS}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Learning rate : {LEARNING_RATE}")

print("\nTensorFlow:", tf.__version__)

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print("\nCSV samples:", len(df))

class_to_index = {
    name: i for i, name in enumerate(CLASSES)
}

# ============================================================
# BUILD DATASET USING EXACT V3 PREPROCESSING
# ============================================================

X = []
y = []

failed = 0

print("\n" + "=" * 70)
print("PROCESSING LANDMARKS")
print("=" * 70)

for i, row in df.iterrows():

    sign = str(row["sign"]).lower()

    if sign not in class_to_index:
        continue

    filename = Path(str(row["path"])).name
    local_path = LANDMARK_DIR / filename

    if not local_path.exists():
        failed += 1
        continue

    try:
        sequence = load_landmarks(local_path)

        if sequence is None or len(sequence) == 0:
            failed += 1
            continue

        sequence = resize_sequence(
            sequence,
            target_length=SEQUENCE_LENGTH
        )

        sequence = normalize_sequence(sequence)

        if sequence.shape != (SEQUENCE_LENGTH, FEATURES):
            failed += 1
            continue

        if np.isnan(sequence).any() or np.isinf(sequence).any():
            failed += 1
            continue

        X.append(sequence.astype(np.float32))
        y.append(class_to_index[sign])

    except Exception as e:
        failed += 1
        print(f"Failed: {filename}: {e}")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)

# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET READY")
print("=" * 70)

print(f"\nX shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"Failed : {failed}")

for i, name in enumerate(CLASSES):
    print(f"{i:2d} {name:10s}: {np.sum(y == i)}")

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

print("\n" + "=" * 70)
print("TRAIN / VALIDATION SPLIT")
print("=" * 70)

print(f"\nTraining samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")

# ============================================================
# CLASS WEIGHTS — SAME IDEA AS V3
# ============================================================

classes_present = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes_present,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes_present, weights)
}

print("\nClass weights:")
for cls, weight in class_weights.items():
    print(f"{CLASSES[cls]:10s}: {weight:.4f}")

# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING 5-CLASS MODEL FROM SCRATCH")
print("=" * 70)

model = Sequential([
    Input(
        shape=(SEQUENCE_LENGTH, FEATURES)
    ),

    Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    ),

    Dropout(0.30),

    Bidirectional(
        LSTM(32)
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
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
]

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
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

    verbose=1
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("5-CLASS V2 MODEL EVALUATION")
print("=" * 70)

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print(f"\nValidation loss     : {loss:.4f}")
print(f"Validation accuracy : {accuracy:.4f}")
print(f"Validation accuracy %: {accuracy * 100:.2f}%")

# ============================================================
# CONFUSION MATRIX
# ============================================================

from sklearn.metrics import confusion_matrix, classification_report

predictions = model.predict(
    X_val,
    verbose=0
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

cm = confusion_matrix(
    y_val,
    predicted_labels,
    labels=np.arange(NUM_CLASSES)
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    "             " +
    " ".join(f"{name:>8}" for name in CLASSES)
)

for i, row in enumerate(cm):
    print(
        f"{CLASSES[i]:10s} " +
        " ".join(f"{value:8d}" for value in row)
    )

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_val,
        predicted_labels,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("5-CLASS V2 TRAINING COMPLETE")
print("=" * 70)

print("\nBest model saved:")
print(MODEL_PATH)

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)

print("\nClasses:")
for i, name in enumerate(CLASSES):
    print(f"{i}: {name}")
