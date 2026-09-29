import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

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


def augment_sequence(seq):
    T = len(seq)

    # 1. Fast speed warp (0.75x - 0.90x)
    f1 = np.random.uniform(0.75, 0.90)
    len1 = max(10, int(T * f1))
    idx1 = np.linspace(0, T - 1, len1)
    warped1 = np.zeros((len1, seq.shape[1]), dtype=np.float32)
    for f in range(seq.shape[1]):
        warped1[:, f] = np.interp(idx1, np.arange(T), seq[:, f])
    resized1 = resize_sequence(warped1, target_length=32)
    norm1 = normalize_sequence(resized1)

    # 2. Slow speed warp (1.10x - 1.25x)
    f2 = np.random.uniform(1.10, 1.25)
    len2 = int(T * f2)
    idx2 = np.linspace(0, T - 1, len2)
    warped2 = np.zeros((len2, seq.shape[1]), dtype=np.float32)
    for f in range(seq.shape[1]):
        warped2[:, f] = np.interp(idx2, np.arange(T), seq[:, f])
    resized2 = resize_sequence(warped2, target_length=32)
    norm2 = normalize_sequence(resized2)

    # 3. Coordinate jitter / sensor noise
    noise = np.random.normal(0, 0.005, size=seq.shape).astype(np.float32)
    mask = (seq != 0).astype(np.float32)
    jittered = (seq + noise * mask).astype(np.float32)
    norm3 = normalize_sequence(jittered)

    return norm1, norm2, norm3


def process_single_class(args):
    sign, sign_idx, file_paths = args
    sequences = []
    skipped = 0

    for p_file in file_paths:
        if p_file.exists() and p_file.stat().st_size > 0:
            try:
                seq = load_landmarks(p_file)
                if len(seq) == 0:
                    skipped += 1
                    continue
                seq_resized = resize_sequence(seq, target_length=32)
                seq_norm = normalize_sequence(seq_resized)
                if seq_norm.shape == (32, 225) and not np.isnan(seq_norm).any():
                    sequences.append(seq_norm)
                    # Add fast, slow, and sensor jitter augmentations
                    w1, w2, j = augment_sequence(seq_norm)
                    if not np.isnan(w1).any():
                        sequences.append(w1)
                    if not np.isnan(w2).any():
                        sequences.append(w2)
                    if not np.isnan(j).any():
                        sequences.append(j)
            except Exception:
                skipped += 1
        else:
            skipped += 1

    # Add camera recorded samples if available (e.g. mom and dad)
    cam_dir = DATA_DIR / "camera_mom_dad" / sign
    if cam_dir.exists():
        for cf in cam_dir.glob("*.npy"):
            try:
                cseq = np.load(cf).astype(np.float32)
                cseq_norm = normalize_sequence(cseq)
                if cseq_norm.shape == (32, 225) and not np.isnan(cseq_norm).any():
                    sequences.append(cseq_norm)
                    # Add augmentations
                    cw1, cw2, cj = augment_sequence(cseq_norm)
                    if not np.isnan(cw1).any():
                        sequences.append(cw1)
                    if not np.isnan(cw2).any():
                        sequences.append(cw2)
                    if not np.isnan(cj).any():
                        sequences.append(cj)
            except Exception:
                pass

    return sign, sign_idx, sequences, skipped


def main():
    print("=" * 70)
    print("SIGNAFLOW AI — BUILDING COMPREHENSIVE 50-CLASS DATASET (75,000+ SAMPLES)")
    print("=" * 70)

    classes = list(np.load(CLASSES_PATH, allow_pickle=True))
    print(f"Target classes count: {len(classes)}")

    df = pd.read_csv(CSV_PATH)
    t0 = time.time()

    # Backup existing X and y if needed
    backup_x = DATA_DIR / "processed" / "X_backup.npy"
    backup_y = DATA_DIR / "processed" / "y_backup.npy"
    if X_PATH.exists() and not backup_x.exists():
        np.save(backup_x, np.load(X_PATH))
        np.save(backup_y, np.load(Y_PATH))
        print(f"Created backup of previous dataset at {backup_x.name}")

    tasks = []
    for sign_idx, sign in enumerate(classes):
        sign_rows = df[df["sign"] == sign]
        file_paths = [PARQUET_DIR / Path(r["path"]).name for _, r in sign_rows.iterrows()]
        tasks.append((sign, sign_idx, file_paths))

    all_sequences = []
    all_labels = []

    print(f"\nProcessing all 50 classes with parallel workers...")
    with ProcessPoolExecutor(max_workers=6) as executor:
        for sign, sign_idx, seqs, skipped in executor.map(process_single_class, tasks):
            print(f"  • {sign:<12} (class {sign_idx:>2}): {len(seqs):>5} samples (skipped: {skipped})")
            for s in seqs:
                all_sequences.append(s)
                all_labels.append(sign_idx)

    X_final = np.asarray(all_sequences, dtype=np.float32)
    y_final = np.asarray(all_labels, dtype=np.int32)

    print("\n" + "=" * 70)
    print("DATASET BUILD COMPLETED!")
    print("=" * 70)
    print(f"Total sequences generated : {len(X_final):,}")
    print(f"X shape                   : {X_final.shape}")
    print(f"y shape                   : {y_final.shape}")
    print(f"Total elapsed time        : {time.time() - t0:.1f}s")
    print("=" * 70)

    # Save
    np.save(X_PATH, X_final)
    np.save(Y_PATH, y_final)
    print("Successfully saved updated X.npy and y.npy!")


if __name__ == "__main__":
    main()
