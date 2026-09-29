import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess_landmarks import (
    load_landmarks,
    resize_sequence,
    normalize_sequence,
)

DATA_DIR = PROJECT_ROOT / "dataset"
PARQUET_DIR = DATA_DIR / "train_landmarks_50"
CSV_PATH = DATA_DIR / "train_50_classes.csv"

X_PATH = DATA_DIR / "processed" / "X.npy"
Y_PATH = DATA_DIR / "processed" / "y.npy"
CLASSES_PATH = DATA_DIR / "processed" / "classes.npy"

CORE_10 = [
    "hello", "mom", "dad", "water", "food",
    "yes", "no", "please", "go", "home"
]


def augment(seq):
    # 1. Speed warp (fast execution 0.75x - 0.90x)
    T = len(seq)
    factor = np.random.uniform(0.75, 0.90)
    new_len = max(10, int(T * factor))
    indices = np.linspace(0, T - 1, new_len)
    warped = np.zeros((new_len, seq.shape[1]), dtype=np.float32)
    for f in range(seq.shape[1]):
        warped[:, f] = np.interp(indices, np.arange(T), seq[:, f])
    resized = resize_sequence(warped, target_length=32)
    norm_warped = normalize_sequence(resized)

    # 2. Scale / jitter (distance variation)
    scale = np.random.uniform(0.94, 1.06)
    noise = np.random.normal(0, 0.004, size=seq.shape).astype(np.float32)
    mask = (seq != 0).astype(np.float32)
    jittered = (seq * scale + noise * mask).astype(np.float32)
    norm_jittered = normalize_sequence(jittered)

    return norm_warped, norm_jittered


def main():
    print("=" * 70)
    print("SIGNAFLOW AI — PROCESSING ALL 3,862 KAGGLE PARQUETS FOR 10 CORE WORDS")
    print("=" * 70)

    classes = list(np.load(CLASSES_PATH, allow_pickle=True))
    df = pd.read_csv(CSV_PATH)

    X_full = np.load(X_PATH)
    y_full = np.load(Y_PATH)

    for sign in CORE_10:
        sign_idx = classes.index(sign)
        sign_rows = df[df["sign"] == sign]
        print(f"\nProcessing '{sign}' ({len(sign_rows)} Kaggle entries)...")

        processed = []
        for _, row in sign_rows.iterrows():
            p_name = Path(row["path"]).name
            p_file = PARQUET_DIR / p_name
            if p_file.exists() and p_file.stat().st_size > 0:
                try:
                    seq = load_landmarks(p_file)
                    if len(seq) == 0:
                        continue
                    seq_resized = resize_sequence(seq, target_length=32)
                    seq_norm = normalize_sequence(seq_resized)
                    if seq_norm.shape == (32, 225) and not np.isnan(seq_norm).any():
                        processed.append(seq_norm)
                        # Generate 2 realistic augmentations
                        w, j = augment(seq_norm)
                        if not np.isnan(w).any():
                            processed.append(w)
                        if not np.isnan(j).any():
                            processed.append(j)
                except Exception:
                    pass

        print(f"Generated {len(processed)} total sequences for '{sign}' (raw + multi-speed augmentations)!")

        # Replace in X and y
        mask_not_sign = (y_full != sign_idx)
        X_other = X_full[mask_not_sign]
        y_other = y_full[mask_not_sign]

        X_new = np.asarray(processed, dtype=np.float32)
        y_new = np.full(len(X_new), sign_idx, dtype=y_full.dtype)

        X_full = np.concatenate([X_other, X_new], axis=0)
        y_full = np.concatenate([y_other, y_new], axis=0)
        print(f"Dataset shape now: {X_full.shape}")

    np.save(X_PATH, X_full)
    np.save(Y_PATH, y_full)
    print("\n" + "=" * 70)
    print("SUCCESS: Full dataset updated and saved to X.npy and y.npy!")
    print("=" * 70)


if __name__ == "__main__":
    main()
