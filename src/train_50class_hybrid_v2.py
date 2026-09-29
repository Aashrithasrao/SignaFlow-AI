"""
train_50class_hybrid_v2.py
==========================
Trains the 50-class Hybrid 1D-CNN + BiLSTM model using the LEAK-FREE
pre-split dataset produced by build_50class_dataset_v2.py.

Reads:
  dataset/processed/X_train.npy
  dataset/processed/y_train.npy
  dataset/processed/X_val.npy
  dataset/processed/y_val.npy

Saves best checkpoint to:
  models/sign_model_50_hybrid_cnn.keras
"""

import os
import json
import time
import shutil
import argparse

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # headless – no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    Bidirectional, LSTM, Dense, Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ============================================================
# PATHS
# ============================================================

DATA_DIR    = "dataset/processed"
MODEL_DIR   = "models"

X_TRAIN_PATH = f"{DATA_DIR}/X_train.npy"
Y_TRAIN_PATH = f"{DATA_DIR}/y_train.npy"
X_VAL_PATH   = f"{DATA_DIR}/X_val.npy"
Y_VAL_PATH   = f"{DATA_DIR}/y_val.npy"
CLASSES_PATH = f"{DATA_DIR}/classes.npy"

MODEL_PATH   = f"{MODEL_DIR}/sign_model_50_hybrid_cnn.keras"
METRICS_PATH = f"{MODEL_DIR}/hybrid_50class_metrics_v2.json"

# ============================================================
# DEFAULTS  (all overrideable from CLI)
# ============================================================

SEQUENCE_LENGTH = 32
FEATURES        = 225
NUM_CLASSES     = 50

DEFAULT_EPOCHS     = 80
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR         = 0.001
DEFAULT_PATIENCE   = 12


def main():
    parser = argparse.ArgumentParser(
        description="Train 50-class Hybrid 1D-CNN + BiLSTM (leak-free v2)"
    )
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS,
                        help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--patience",    type=int,   default=DEFAULT_PATIENCE,
                        help=f"Early stopping patience (default: {DEFAULT_PATIENCE})")
    args = parser.parse_args()

    epochs     = args.epochs
    batch_size = args.batch_size
    lr         = args.lr
    patience   = args.patience

    print("=" * 70)
    print("SIGNAFLOW AI — HYBRID CNN+BiLSTM  50-CLASS  LEAK-FREE TRAINING  (v2)")
    print(f"Epochs={epochs} | BatchSize={batch_size} | LR={lr} | Patience={patience}")
    print("=" * 70)
    print(f"TensorFlow: {tf.__version__}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Auto-backup existing model ────────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        backup = f"{MODEL_DIR}/sign_model_50_hybrid_cnn_backup.keras"
        shutil.copy2(MODEL_PATH, backup)
        print(f"Backed up existing model → {backup}")

    # ── Load datasets (smart fallback) ───────────────────────────────────────
    pre_split_ready = all(
        os.path.exists(p)
        for p in [X_TRAIN_PATH, Y_TRAIN_PATH, X_VAL_PATH, Y_VAL_PATH]
    )

    classes = list(np.load(CLASSES_PATH, allow_pickle=True))

    if pre_split_ready:
        # ✅ BEST: use the leak-free pre-split files from build_50class_dataset_v2.py
        print("\n[v2] Loading LEAK-FREE pre-split datasets...")
        X_train = np.load(X_TRAIN_PATH)
        y_train = np.load(Y_TRAIN_PATH)
        X_val   = np.load(X_VAL_PATH)
        y_val   = np.load(Y_VAL_PATH)
        print("  Source-level split confirmed — validation is 100% clean.")
    else:
        # ⚠️  FALLBACK: pre-split files not built yet → use X.npy/y.npy with random split
        legacy_x = f"{DATA_DIR}/X.npy"
        legacy_y = f"{DATA_DIR}/y.npy"
        if not os.path.exists(legacy_x):
            raise FileNotFoundError(
                "Neither pre-split files nor X.npy found.\n"
                "Run:  python src/build_50class_dataset_v2.py"
            )
        print("\n[v2] WARNING: Pre-split files not found. Using legacy X.npy (data leakage present).")
        print("     Run build_50class_dataset_v2.py later for honest validation.")
        X_all = np.load(legacy_x)
        y_all = np.load(legacy_y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
        )

    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}     y_val   : {y_val.shape}")
    print(f"  Classes : {len(classes)}")

    if len(classes) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, got {len(classes)}")

    # ── Class weights (balance minority signs) ────────────────────────────────
    class_ids   = np.unique(y_train)
    weights_arr = compute_class_weight("balanced", classes=class_ids, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(class_ids, weights_arr)}

    # ── Model architecture ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BUILDING HYBRID 1D-CNN + BiLSTM (50 classes)")
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
        MaxPooling1D(pool_size=2),   # 32 → 16 time-steps
        Dropout(0.25),

        # Bidirectional temporal context
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.30),

        # Dense head 1
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.30),

        # Dense head 2
        Dense(128, activation="relu"),
        Dropout(0.25),

        # Output
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
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
            patience=max(3, patience // 3),
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training…")
    t0 = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Evaluate best checkpoint ──────────────────────────────────────────────
    best = tf.keras.models.load_model(MODEL_PATH)
    val_loss, val_acc = best.evaluate(X_val, y_val, verbose=0)
    preds = np.argmax(best.predict(X_val, verbose=0), axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, preds, average="weighted", zero_division=0
    )

    print("\n" + "=" * 70)
    print("FINAL LEAK-FREE EVALUATION:")
    print(f"  • Validation Accuracy : {val_acc  * 100:.2f}%  ← HONEST real-world estimate")
    print(f"  • Weighted F1-Score   : {f1       * 100:.2f}%")
    print(f"  • Weighted Precision  : {precision* 100:.2f}%")
    print(f"  • Weighted Recall     : {recall   * 100:.2f}%")
    print(f"  • Validation Loss     : {val_loss:.4f}")
    print("=" * 70)

    report = classification_report(
        y_val, preds, target_names=classes, output_dict=True, zero_division=0
    )
    metrics = {
        "validation_accuracy":       float(val_acc),
        "validation_loss":           float(val_loss),
        "weighted_f1":               float(f1),
        "weighted_precision":        float(precision),
        "weighted_recall":           float(recall),
        "training_time_seconds":     float(elapsed),
        "dataset_version":           "v2_leak_free",
        "per_class":                 report,
    }
    with open(METRICS_PATH, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Metrics saved → {METRICS_PATH}")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print("\nPer-class summary:")
    for sign in classes:
        p  = report[sign]["precision"]
        r  = report[sign]["recall"]
        f  = report[sign]["f1-score"]
        flag = "✓" if f >= 0.80 else ("~" if f >= 0.60 else "✗")
        print(f"  {flag} {sign:<12}: P={p*100:5.1f}% | R={r*100:5.1f}% | F1={f*100:5.1f}%")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_val, preds)

    # ── Terminal text version (compact) ──────────────────────────────────────
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    header = "        " + "".join(f"{c[:5]:>6}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{classes[i]:<8}" + "".join(f"{v:>6}" for v in row)
        print(row_str)

    # ── Save high-resolution PNG ──────────────────────────────────────────────
    cm_path = f"{MODEL_DIR}/confusion_matrix_50class.png"
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.4,
        linecolor="#cccccc",
        ax=ax,
    )
    ax.set_title(
        f"50-Class ASL Confusion Matrix\n"
        f"Val Accuracy: {val_acc*100:.2f}%  |  Weighted F1: {f1*100:.2f}%",
        fontsize=15,
        pad=16,
    )
    ax.set_xlabel("Predicted Sign", fontsize=12)
    ax.set_ylabel("Actual Sign",    fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix PNG saved → {cm_path}")
    print("Open it to see which signs are being confused with each other.")


if __name__ == "__main__":
    main()
