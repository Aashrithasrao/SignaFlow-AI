from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

from src.preprocess_landmarks import (
    load_landmarks,
    resize_sequence,
    normalize_sequence,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "dataset" / "train_5_classes.csv"
LANDMARK_DIR = PROJECT_ROOT / "dataset" / "train_landmarks_50"
NO_SIGN_DIR = PROJECT_ROOT / "dataset" / "no_sign"

MODEL_PATH = PROJECT_ROOT / "models" / "sign_activity_gate_v1.keras"

SEQUENCE_LENGTH = 32
FEATURES = 225

EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.0005

MAX_NO_SIGN = 650

print("=" * 70)
print("SIGNAFLOW AI — SIGN ACTIVITY GATE")
print("=" * 70)

# ============================================================
# LOAD REAL SIGN DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

if "sign" in df.columns:
    label_col = "sign"
elif "label" in df.columns:
    label_col = "label"
else:
    raise ValueError(
        f"Cannot find label column: {list(df.columns)}"
    )

if "sequence_id" in df.columns:
    id_col = "sequence_id"
elif "id" in df.columns:
    id_col = "id"
else:
    raise ValueError(
        f"Cannot find sequence ID column: {list(df.columns)}"
    )

parquet_files = {
    p.stem: p
    for p in LANDMARK_DIR.rglob("*.parquet")
}

X_sign = []

print("\nLoading SIGN samples...")

for _, row in df.iterrows():

    sequence_id = str(row[id_col]).strip()

    path = parquet_files.get(sequence_id)

    if path is None:
        continue

    try:

        sequence = load_landmarks(path)

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
            continue

        X_sign.append(
            sequence.astype(np.float32)
        )

    except Exception:
        continue

X_sign = np.asarray(
    X_sign,
    dtype=np.float32
)

# 1 = SIGN
y_sign = np.ones(
    len(X_sign),
    dtype=np.int64
)

print(
    "SIGN samples:",
    len(X_sign)
)

# ============================================================
# LOAD NO_SIGN
# ============================================================

print("\nLoading NO_SIGN samples...")

no_sign_files = sorted(
    NO_SIGN_DIR.glob("*.npy")
)

# Use all currently collected samples,
# but cap at MAX_NO_SIGN.
if len(no_sign_files) > MAX_NO_SIGN:
    no_sign_files = no_sign_files[:MAX_NO_SIGN]

X_idle = []

for path in no_sign_files:

    try:

        sequence = np.load(
            path
        ).astype(np.float32)

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
            continue

        X_idle.append(
            sequence.astype(np.float32)
        )

    except Exception:
        continue

X_idle = np.asarray(
    X_idle,
    dtype=np.float32
)

# 0 = NO_SIGN
y_idle = np.zeros(
    len(X_idle),
    dtype=np.int64
)

print(
    "NO_SIGN samples:",
    len(X_idle)
)

# ============================================================
# BALANCE
# ============================================================

sign_count = len(X_sign)
idle_count = len(X_idle)

target = min(
    sign_count,
    idle_count
)

rng = np.random.default_rng(42)

sign_indices = rng.choice(
    sign_count,
    target,
    replace=False
)

idle_indices = rng.choice(
    idle_count,
    target,
    replace=False
)

X_sign = X_sign[
    sign_indices
]

y_sign = y_sign[
    sign_indices
]

X_idle = X_idle[
    idle_indices
]

y_idle = y_idle[
    idle_indices
]

# ============================================================
# COMBINE
# ============================================================

X = np.concatenate(
    [
        X_idle,
        X_sign
    ],
    axis=0
)

y = np.concatenate(
    [
        y_idle,
        y_sign
    ],
    axis=0
)

# Shuffle
indices = rng.permutation(
    len(X)
)

X = X[
    indices
]

y = y[
    indices
]

print("\n" + "=" * 70)
print("GATE DATASET")
print("=" * 70)

print("X shape:", X.shape)
print("y shape:", y.shape)

print(
    "NO_SIGN:",
    np.sum(y == 0)
)

print(
    "SIGN:",
    np.sum(y == 1)
)

# ============================================================
# SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining:", len(X_train))
print("Validation:", len(X_val))

# ============================================================
# CLASS WEIGHTS
# ============================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    int(c): float(w)
    for c, w in zip(
        np.unique(y_train),
        weights
    )
}

# ============================================================
# BINARY MODEL
# ============================================================

model = tf.keras.Sequential([

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

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            32
        )
    ),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING ACTIVITY GATE TRAINING")
print("=" * 70)

model.fit(
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
    verbose=1
)

# ============================================================
# EVALUATE
# ============================================================

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

results = best_model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print("\n" + "=" * 70)
print("ACTIVITY GATE EVALUATION")
print("=" * 70)

print(
    f"Validation loss     : {results[0]:.4f}"
)

print(
    f"Validation accuracy : {results[1]:.4f}"
)

print(
    f"Validation accuracy %: {results[1] * 100:.2f}%"
)

print(
    f"Precision           : {results[2]:.4f}"
)

print(
    f"Recall              : {results[3]:.4f}"
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

probabilities = best_model.predict(
    X_val,
    verbose=0
).ravel()

predictions = (
    probabilities >= 0.5
).astype(int)

cm = confusion_matrix(
    y_val,
    predictions
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "              NO_SIGN   SIGN"
)

print(
    f"NO_SIGN      {cm[0,0]:8d} {cm[0,1]:7d}"
)

print(
    f"SIGN         {cm[1,0]:8d} {cm[1,1]:7d}"
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_val,
        predictions,
        target_names=[
            "no_sign",
            "sign"
        ],
        digits=4,
        zero_division=0
    )
)

print("\n" + "=" * 70)
print("ACTIVITY GATE TRAINING COMPLETE")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)
