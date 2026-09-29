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

MODEL_PATH = PROJECT_ROOT / "models" / "sign_model_6class_v1.keras"

CLASSES = [
    "hello",
    "mom",
    "dad",
    "water",
    "food",
    "no_sign",
]

SEQUENCE_LENGTH = 32
FEATURES = 225

EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 0.0005

print("=" * 70)
print("SIGNAFLOW AI — 6 CLASS TRAINING")
print("=" * 70)

# ============================================================
# LOAD 5 SIGN CLASSES
# ============================================================

print("\nLoading 5 sign classes...")

df = pd.read_csv(CSV_PATH)

# Find label column
if "sign" in df.columns:
    label_col = "sign"
elif "label" in df.columns:
    label_col = "label"
else:
    raise ValueError(
        f"Could not find sign/label column. Columns: {list(df.columns)}"
    )

# Find sequence ID column
if "sequence_id" in df.columns:
    id_col = "sequence_id"
elif "id" in df.columns:
    id_col = "id"
else:
    raise ValueError(
        f"Could not find sequence_id/id column. Columns: {list(df.columns)}"
    )

print(f"Label column: {label_col}")
print(f"Sequence column: {id_col}")

parquet_files = {
    p.stem: p
    for p in LANDMARK_DIR.rglob("*.parquet")
}

X_sign = []
y_sign = []

class_to_index = {
    name: i
    for i, name in enumerate(CLASSES[:5])
}

for count, (_, row) in enumerate(df.iterrows(), 1):

    label = str(row[label_col]).strip().lower()

    if label not in class_to_index:
        continue

    sequence_id = str(row[id_col]).strip()

    path = parquet_files.get(sequence_id)

    if path is None:
        continue

    try:
        sequence = load_landmarks(path)

        sequence = resize_sequence(
            sequence,
            SEQUENCE_LENGTH,
        )

        sequence = normalize_sequence(
            sequence,
        )

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURES,
        ):
            continue

        X_sign.append(sequence.astype(np.float32))
        y_sign.append(class_to_index[label])

    except Exception as e:
        print(
            f"\nSkipping {sequence_id}: {e}"
        )

    if count % 200 == 0:
        print(
            f"Processed {count}/{len(df)}",
            end="\r",
            flush=True,
        )

print("\n")

X_sign = np.asarray(
    X_sign,
    dtype=np.float32,
)

y_sign = np.asarray(
    y_sign,
    dtype=np.int64,
)

print("5-class samples loaded:", len(X_sign))

for name, index in class_to_index.items():
    print(
        f"  {name:8s}: "
        f"{np.sum(y_sign == index)}"
    )

# ============================================================
# LOAD NO_SIGN
# ============================================================

print("\nLoading NO_SIGN samples...")

no_sign_files = sorted(
    NO_SIGN_DIR.glob("no_sign_*.npy")
)

if not no_sign_files:
    raise RuntimeError(
        "No NO_SIGN samples found."
    )

X_idle = []

for path in no_sign_files:

    try:

        sequence = np.load(
            path
        ).astype(np.float32)

        sequence = resize_sequence(
            sequence,
            SEQUENCE_LENGTH,
        )

        sequence = normalize_sequence(
            sequence,
        )

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURES,
        ):
            continue

        X_idle.append(
            sequence.astype(np.float32)
        )

    except Exception as e:
        print(
            f"Skipping {path.name}: {e}"
        )

X_idle = np.asarray(
    X_idle,
    dtype=np.float32,
)

y_idle = np.full(
    len(X_idle),
    5,
    dtype=np.int64,
)

print(
    "NO_SIGN samples loaded:",
    len(X_idle),
)

# ============================================================
# COMBINE
# ============================================================

X = np.concatenate(
    [
        X_sign,
        X_idle,
    ],
    axis=0,
)

y = np.concatenate(
    [
        y_sign,
        y_idle,
    ],
    axis=0,
)

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", len(CLASSES))

for i, name in enumerate(CLASSES):
    print(
        f"  {i}: {name:8s} "
        f"{np.sum(y == i)} samples"
    )

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))

# ============================================================
# CLASS WEIGHTS
# ============================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(
        np.unique(y_train),
        weights,
    )
}

print("\nClass weights:")

for i, name in enumerate(CLASSES):
    print(
        f"  {name:8s}: "
        f"{class_weights[i]:.3f}"
    )

# ============================================================
# MODEL
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURES,
        )
    ),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,
            return_sequences=True,
        )
    ),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            32,
        )
    ),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        64,
        activation="relu",
    ),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        len(CLASSES),
        activation="softmax",
    ),
])

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

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

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING 6-CLASS TRAINING")
print("=" * 70)

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val,
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

print("\nLoading best model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

loss, accuracy = best_model.evaluate(
    X_val,
    y_val,
    verbose=0,
)

print("\n" + "=" * 70)
print("6-CLASS MODEL EVALUATION")
print("=" * 70)

print(
    f"Validation loss     : {loss:.4f}"
)

print(
    f"Validation accuracy : {accuracy:.4f}"
)

print(
    f"Validation accuracy %: {accuracy * 100:.2f}%"
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

probabilities = best_model.predict(
    X_val,
    verbose=0,
)

predictions = np.argmax(
    probabilities,
    axis=1,
)

cm = confusion_matrix(
    y_val,
    predictions,
    labels=np.arange(len(CLASSES)),
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("Rows = Actual")
print("Columns = Predicted")
print()

print(
    "          "
    + " ".join(
        f"{name[:8]:>8}"
        for name in CLASSES
    )
)

for i, name in enumerate(CLASSES):

    print(
        f"{name[:8]:>8} "
        + " ".join(
            f"{cm[i, j]:8d}"
            for j in range(len(CLASSES))
        )
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
        predictions,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )
)

print("\n" + "=" * 70)
print("6-CLASS TRAINING COMPLETE")
print("=" * 70)

print("\nBest model:")
print(MODEL_PATH)

print("\nClasses:")

for i, name in enumerate(CLASSES):
    print(f"{i}: {name}")
