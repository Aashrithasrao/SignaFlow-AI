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
CSV_PATH = DATA_DIR / "train.csv"
CAMERA_NO_DIR = DATA_DIR / "camera_no" / "no"

X_PATH = DATA_DIR / "processed" / "X.npy"
Y_PATH = DATA_DIR / "processed" / "y.npy"
CLASSES_PATH = DATA_DIR / "processed" / "classes.npy"


def main():
    print("=" * 70)
    print("SIGNAFLOW AI — UPDATE 'NO' SAMPLES FROM DOWNLOADED PARQUETS & WEBCAM")
    print("=" * 70)

    classes = list(np.load(CLASSES_PATH, allow_pickle=True))
    no_class_idx = classes.index("no")
    print(f"Target class 'no' index: {no_class_idx}")

    df = pd.read_csv(CSV_PATH)
    no_rows = df[df["sign"] == "no"]
    print(f"Total 'no' entries in CSV: {len(no_rows)}")

    # 1. Process all available parquets for 'no' in PARQUET_DIR
    processed_no_sequences = []
    skipped = 0

    for idx, row in no_rows.iterrows():
        p_name = Path(row["path"]).name
        p_file = PARQUET_DIR / p_name
        if p_file.exists() and p_file.stat().st_size > 0:
            try:
                seq = load_landmarks(p_file)
                if len(seq) == 0:
                    skipped += 1
                    continue
                seq_resized = resize_sequence(seq, target_length=32)
                seq_norm = normalize_sequence(seq_resized)
                if seq_norm.shape == (32, 225) and not np.isnan(seq_norm).any():
                    processed_no_sequences.append(seq_norm)
            except Exception as e:
                skipped += 1

    print(f"Successfully processed {len(processed_no_sequences)} 'no' sequences from Kaggle parquets (skipped {skipped})")

    # Generate speed and temporal augmentations to match real-time webcam speeds
    print("Generating temporal and spatial augmentations for 'no'...")
    augmented_sequences = []
    for seq in processed_no_sequences:
        # Augmentation 1: Fast signing speed (time warp 0.75x - 0.90x)
        T = len(seq)
        factor = np.random.uniform(0.75, 0.90)
        new_len = max(10, int(T * factor))
        indices = np.linspace(0, T - 1, new_len)
        warped = np.zeros((new_len, seq.shape[1]), dtype=np.float32)
        for f in range(seq.shape[1]):
            warped[:, f] = np.interp(indices, np.arange(T), seq[:, f])
        resized = resize_sequence(warped, target_length=32)
        norm_warped = normalize_sequence(resized)
        if not np.isnan(norm_warped).any():
            augmented_sequences.append(norm_warped)

        # Augmentation 2: Spatial scale and subtle jitter (webcam distance variation)
        scale = np.random.uniform(0.94, 1.06)
        noise = np.random.normal(0, 0.004, size=seq.shape).astype(np.float32)
        mask = (seq != 0).astype(np.float32)
        jittered = (seq * scale + noise * mask).astype(np.float32)
        norm_jittered = normalize_sequence(jittered)
        if not np.isnan(norm_jittered).any():
            augmented_sequences.append(norm_jittered)

    processed_no_sequences.extend(augmented_sequences)
    print(f"Total 'no' sequences after multi-speed augmentations: {len(processed_no_sequences)}")

    # 2. Add webcam captured samples if any
    if CAMERA_NO_DIR.exists():
        cam_files = list(CAMERA_NO_DIR.glob("*.npy"))
        cam_added = 0
        for f in cam_files:
            try:
                arr = np.load(f)
                if arr.shape == (32, 225) and not np.isnan(arr).any():
                    processed_no_sequences.append(arr)
                    cam_added += 1
            except Exception:
                pass
        print(f"Added {cam_added} camera-captured 'no' sequences from {CAMERA_NO_DIR}")

    # 3. Load existing X.npy and y.npy
    X_full = np.load(X_PATH)
    y_full = np.load(Y_PATH)
    print(f"\nExisting X shape: {X_full.shape} (currently {np.sum(y_full == no_class_idx)} 'no' samples)")

    # Replace old 'no' samples in X and y with the expanded set
    mask_not_no = (y_full != no_class_idx)
    X_other = X_full[mask_not_no]
    y_other = y_full[mask_not_no]

    X_no_new = np.asarray(processed_no_sequences, dtype=np.float32)
    y_no_new = np.full(len(X_no_new), no_class_idx, dtype=y_full.dtype)

    X_updated = np.concatenate([X_other, X_no_new], axis=0)
    y_updated = np.concatenate([y_other, y_no_new], axis=0)

    print(f"Updated X shape: {X_updated.shape} (now {len(X_no_new)} 'no' samples!)")

    # Save backup first
    backup_x = DATA_DIR / "processed" / "X_backup.npy"
    backup_y = DATA_DIR / "processed" / "y_backup.npy"
    if not backup_x.exists():
        np.save(backup_x, X_full)
        np.save(backup_y, y_full)
        print(f"Created backup at {backup_x.name}")

    # Save updated
    np.save(X_PATH, X_updated)
    np.save(Y_PATH, y_updated)
    print("Saved updated X.npy and y.npy successfully!")


if __name__ == "__main__":
    main()
