import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "dataset/train_5_classes.csv"
LANDMARK_DIR = Path("dataset/train_landmarks_50")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "sign_model_5class_v1.keras"

SEQUENCE_LENGTH = 32
FEATURES = 225
NUM_CLASSES = 5

CLASSES = [
    "hello",
    "mom",
    "dad",
    "water",
    "food",
]

EPOCHS = 80
BATCH_SIZE = 32
RANDOM_STATE = 42

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("=" * 70)
print("SIGNALFLOW AI - 5 CLASS MODEL TRAINING FROM SCRATCH")
print("=" * 70)

print("\nTensorFlow:", tf.__version__)
print("Classes:", CLASSES)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print("\nCSV samples:", len(df))

# ============================================================
# LANDMARK LOADER
# ============================================================

def load_landmarks(path):
    """
    Load a parquet landmark sequence and convert it to
    exactly (32, 225).
    """

    data = pd.read_parquet(path)

    # Convert dataframe to numpy
    arr = data.select_dtypes(include=[np.number]).to_numpy(
        dtype=np.float32
    )

    if arr.size == 0:
        return None

    # Flatten if necessary
    if arr.ndim == 1:
        arr = arr.reshape(-1, FEATURES)

    # If there are unexpected feature dimensions,
    # keep only the first 225.
    if arr.shape[1] > FEATURES:
        arr = arr[:, :FEATURES]

    # If fewer than 225 features, pad with zeros
    elif arr.shape[1] < FEATURES:
        padded = np.zeros(
            (arr.shape[0], FEATURES),
            dtype=np.float32
        )
        padded[:, :arr.shape[1]] = arr
        arr = padded

    # --------------------------------------------------------
    # Temporal resize to 32 frames
    # --------------------------------------------------------

    frames = arr.shape[0]

    if frames == 0:
        return None

    if frames == SEQUENCE_LENGTH:
        result = arr

    else:
        indices = np.linspace(
            0,
            frames - 1,
            SEQUENCE_LENGTH
        ).astype(int)

        result = arr[indices]

    # --------------------------------------------------------
    # Normalize each sequence
    # --------------------------------------------------------

    mean = np.mean(result, axis=0, keepdims=True)
    std = np.std(result, axis=0, keepdims=True)

    result = (result - mean) / (std + 1e-6)

    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return result.astype(np.float32)


# ============================================================
# BUILD DATASET
# ============================================================

X = []
y = []

mapping = {
    name: i
    for i, name in enumerate(CLASSES)
}

failed = 0

print("\nLoading landmark sequences...\n")

for index, row in df.iterrows():

    path = str(row["path"])
    filename = Path(path).name

    local_path = LANDMARK_DIR / filename

    if not local_path.exists():
        failed += 1
        continue

    try:
        sequence = load_landmarks(local_path)

        if sequence is None:
            failed += 1
            continue

        if sequence.shape != (SEQUENCE_LENGTH, FEATURES):
            failed += 1
            continue

        label = mapping[str(row["sign"]).lower()]

        X.append(sequence)
        y.append(label)

    except Exception as e:
        failed += 1
        print(
            f"Failed: {filename} -> {e}"
        )

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)

print("\n" + "=" * 70)
print("DATASET READY")
print("=" * 70)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Failed:", failed)

for i, name in enumerate(CLASSES):
    print(
        f"{name:10} : {np.sum(y == i)}"
    )

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))

# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MODEL FROM SCRATCH")
print("=" * 70)

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(SEQUENCE_LENGTH, FEATURES)
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
        NUM_CLASSES,
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
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
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
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

    shuffle=True,

    callbacks=callbacks,

    verbose=1
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("5-CLASS MODEL EVALUATION")
print("=" * 70)

loss, accuracy = best_model.evaluate(
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

predictions = best_model.predict(
    X_val,
    verbose=0
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_val,
    predicted_labels
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    "             " +
    " ".join(f"{x:>8}" for x in CLASSES)
)

for i, row in enumerate(cm):
    print(
        f"{CLASSES[i]:10} " +
        " ".join(f"{x:8}" for x in row)
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
        digits=4
    )
)

# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("5-CLASS TRAINING COMPLETE")
print("=" * 70)

print("\nBest model:")
print(MODEL_PATH)

print("\nClasses:")
for i, name in enumerate(CLASSES):
    print(f"{i}: {name}")

print("\nModel input:")
print(best_model.input_shape)

print("\nModel output:")
print(best_model.output_shape)

print("\nNext step:")
print("Test the 5-class model with the live camera.")
