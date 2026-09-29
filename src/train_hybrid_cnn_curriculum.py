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

MODEL_5CLASS_PATH = f"{MODEL_DIR}/sign_model_5class_hybrid_cnn.keras"
CLASSES_5CLASS_PATH = f"{MODEL_DIR}/classes_5class.npy"

MODEL_10CLASS_PATH = f"{MODEL_DIR}/sign_model_10class_hybrid_cnn.keras"
CLASSES_10CLASS_PATH = f"{MODEL_DIR}/classes_10class.npy"

MODEL_15CLASS_PATH = f"{MODEL_DIR}/sign_model_15class_hybrid_cnn.keras"
CLASSES_15CLASS_PATH = f"{MODEL_DIR}/classes_15class.npy"

SEQUENCE_LENGTH = 32
FEATURES = 225

TARGET_5_CLASSES = ["hello", "mom", "dad", "water", "food"]
TARGET_10_CLASSES = [
    "hello", "mom", "dad", "water", "food",
    "yes", "no", "please", "go", "home"
]
TARGET_15_CLASSES = [
    "hello", "mom", "dad", "water", "food",
    "yes", "no", "please", "go", "home",
    "thankyou", "drink", "happy", "sad", "hungry"
]

print("=" * 70)
print("SIGNAFLOW AI — INCREMENTAL HYBRID 1D-CNN TRAINING")
print("=" * 70)

# Load full processed dataset
print("\nLoading processed dataset...")
X_full = np.load(X_PATH)
y_full = np.load(Y_PATH)
classes_full = np.load(CLASSES_PATH, allow_pickle=True)
class_to_idx = {c: i for i, c in enumerate(classes_full)}


def train_subset_model(target_classes, model_save_path, classes_save_path, epochs=45):
    print("\n" + "=" * 70)
    print(f"TRAINING HYBRID 1D-CNN FOR {len(target_classes)} CLASSES: {target_classes}")
    print("=" * 70)

    # Filter data
    target_indices = [class_to_idx[c] for c in target_classes]
    mask = np.isin(y_full, target_indices)

    X_subset = X_full[mask]
    y_raw = y_full[mask]

    mapping = {old: new for new, old in enumerate(target_indices)}
    y_subset = np.array([mapping[val] for val in y_raw], dtype=np.int32)

    # Save classes array
    np.save(classes_save_path, np.array(target_classes))
    print(f"Saved class list to: {classes_save_path}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_subset, y_subset, test_size=0.20, random_state=42, stratify=y_subset
    )

    print(f"Total samples      : {len(X_subset):,}")
    print(f"Training samples   : {len(X_train):,}")
    print(f"Validation samples : {len(X_val):,}")

    # Class distribution
    for i, name in enumerate(target_classes):
        print(f"  {i}: {name:<12} (train: {np.sum(y_train == i)}, val: {np.sum(y_val == i)})")

    # Compute balanced weights
    class_weights_array = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = {int(c): float(w) for c, w in zip(np.unique(y_train), class_weights_array)}

    # Build model
    num_classes = len(target_classes)
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, FEATURES)),
        Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        Dropout(0.20),
        Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.30),
        Dense(64 if num_classes <= 5 else 128, activation="relu"),
        Dropout(0.25),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
    ]

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")

    # Evaluate best model
    best_model = tf.keras.models.load_model(model_save_path)
    loss, acc = best_model.evaluate(X_val, y_val, verbose=0)
    preds = np.argmax(best_model.predict(X_val, verbose=0), axis=1)

    p, r, f1, _ = precision_recall_fscore_support(y_val, preds, average="weighted", zero_division=0)

    print("\n" + "-" * 60)
    print(f"RESULTS FOR {num_classes}-CLASS MODEL:")
    print(f"  • Validation Accuracy : {acc * 100:.2f}%")
    print(f"  • Weighted F1-Score   : {f1 * 100:.2f}%")
    print(f"  • Validation Loss     : {loss:.4f}")
    print("-" * 60)

    print("\nDetailed Per-Class Performance:")
    report = classification_report(y_val, preds, target_names=target_classes, zero_division=0)
    print(report)

    return best_model, acc


if __name__ == "__main__":
    import sys

    # 1. 5-Class model (already trained and saved, retrain only if requested)
    if "--all" in sys.argv or not os.path.exists(MODEL_5CLASS_PATH):
        train_subset_model(
            TARGET_5_CLASSES,
            MODEL_5CLASS_PATH,
            CLASSES_5CLASS_PATH,
            epochs=40
        )
    else:
        print(f"\n[INFO]: 5-Class model already exists at {MODEL_5CLASS_PATH} (skipping re-training)")

    # 2. 10-Class model (retrain on newly expanded NO dataset)
    train_subset_model(
        TARGET_10_CLASSES,
        MODEL_10CLASS_PATH,
        CLASSES_10CLASS_PATH,
        epochs=45
    )

    # 3. 15-Class model (retrain on newly expanded NO dataset)
    train_subset_model(
        TARGET_15_CLASSES,
        MODEL_15CLASS_PATH,
        CLASSES_15CLASS_PATH,
        epochs=45
    )

