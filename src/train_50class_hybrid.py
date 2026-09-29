import os
import json
import time
import argparse
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
METRICS_PATH = f"{MODEL_DIR}/hybrid_50class_metrics.json"

SEQUENCE_LENGTH = 32
FEATURES = 225
NUM_CLASSES = 50

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

# Default hyper-parameters (can also be overridden via command-line flags)
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 12


def main():
    parser = argparse.ArgumentParser(description="Train 50-class Hybrid 1D-CNN + BiLSTM model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--patience", type=int, default=PATIENCE, help=f"Early stopping patience (default: {PATIENCE})")
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    patience = args.patience

    print("=" * 70)
    print("SIGNAFLOW AI — TRAINING 50-CLASS HIGH ACCURACY HYBRID 1D-CNN + BiLSTM")
    print(f"Hyperparameters: Epochs={epochs} | BatchSize={batch_size} | LR={lr} | Patience={patience}")
    print("=" * 70)

    print(f"TensorFlow Version: {tf.__version__}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Safely back up existing model before training
    if os.path.exists(MODEL_PATH):
        backup_path = f"{MODEL_DIR}/sign_model_50_hybrid_cnn_backup.keras"
        import shutil
        shutil.copy2(MODEL_PATH, backup_path)
        print(f"Backed up existing hybrid model to: {backup_path}")

    print("\nLoading dataset...")
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    classes = list(np.load(CLASSES_PATH, allow_pickle=True))

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {len(classes)}")

    if len(classes) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, got {len(classes)}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\nTraining samples   : {len(X_train):,}")
    print(f"Validation samples : {len(X_val):,}")

    # Class weights
    class_ids = np.unique(y_train)
    weights_arr = compute_class_weight("balanced", classes=class_ids, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(class_ids, weights_arr)}

    # Build scaled model architecture
    print("\n" + "=" * 70)
    print("BUILDING DEEP HYBRID 1D-CNN + BiLSTM MODEL (50 CLASSES)")
    print("=" * 70)

    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, FEATURES)),

        # Conv Block 1
        Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        Dropout(0.20),

        # Conv Block 2
        Conv1D(256, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),  # 32 -> 16 time steps
        Dropout(0.25),

        # Bidirectional Recurrent Context
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.30),

        # Dense Head 1
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.30),

        # Dense Head 2
        Dense(128, activation="relu"),
        Dropout(0.25),

        # Output
        Dense(NUM_CLASSES, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, patience // 3),
            min_lr=1e-5,
            verbose=1
        ),
    ]

    print("\nStarting model training...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Evaluate best model
    best_model = tf.keras.models.load_model(MODEL_PATH)
    val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
    preds = np.argmax(best_model.predict(X_val, verbose=0), axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, preds, average="weighted", zero_division=0
    )

    print("\n" + "=" * 70)
    print("FINAL 50-CLASS EVALUATION RESULTS:")
    print(f"  • Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"  • Weighted F1-Score   : {f1 * 100:.2f}%")
    print(f"  • Weighted Precision  : {precision * 100:.2f}%")
    print(f"  • Weighted Recall     : {recall * 100:.2f}%")
    print(f"  • Validation Loss     : {val_loss:.4f}")
    print("=" * 70)

    report_dict = classification_report(y_val, preds, target_names=classes, output_dict=True, zero_division=0)
    metrics_summary = {
        "validation_accuracy": float(val_acc),
        "validation_loss": float(val_loss),
        "weighted_f1": float(f1),
        "training_time_seconds": float(elapsed),
        "per_class": report_dict
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved evaluation metrics to {METRICS_PATH}")

    # Print top-performing classes
    print("\nPer-class summary:")
    for sign in classes:
        p = report_dict[sign]["precision"]
        r = report_dict[sign]["recall"]
        f_score = report_dict[sign]["f1-score"]
        print(f"  {sign:<12}: Precision={p*100:5.1f}% | Recall={r*100:5.1f}% | F1={f_score*100:5.1f}%")


if __name__ == "__main__":
    main()
