"""
build_50class_dataset_v2.py
===========================
Builds a LEAK-FREE 50-class ASL dataset by performing the train/val split
at the SOURCE FILE level BEFORE any augmentation is applied.

TRAINING SET  : original + 3 augmented variants (4x multiplier)
VALIDATION SET: ONLY the original clean sequence (no augmentation)

This ensures the model's validation accuracy is an honest measure of
real-world, never-seen-before performance.

Output files:
  dataset/processed/X_train.npy  — training sequences  (shape: N_train, 32, 225)
  dataset/processed/y_train.npy  — training labels     (shape: N_train,)
  dataset/processed/X_val.npy    — validation sequences (shape: N_val,   32, 225)
  dataset/processed/y_val.npy    — validation labels    (shape: N_val,)
  dataset/processed/classes.npy  — 50 class name array  (unchanged)
"""

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

# ============================================================
# PATHS
# ============================================================

DATA_DIR    = PROJECT_ROOT / "dataset"
PARQUET_DIR = DATA_DIR / "train_landmarks_50"
CSV_PATH    = DATA_DIR / "train_50_classes.csv"
PROC_DIR    = DATA_DIR / "processed"
CLASSES_PATH = PROC_DIR / "classes.npy"

X_TRAIN_PATH = PROC_DIR / "X_train.npy"
Y_TRAIN_PATH = PROC_DIR / "y_train.npy"
X_VAL_PATH   = PROC_DIR / "X_val.npy"
Y_VAL_PATH   = PROC_DIR / "y_val.npy"

# ============================================================
# CONSTANTS
# ============================================================

TARGET_LEN   = 32
VAL_SPLIT    = 0.20   # 20% of source files held out for validation
RANDOM_SEED  = 42


# ============================================================
# AUGMENTATION  (applied ONLY to training samples)
# ============================================================

def augment_sequence(seq):
    """
    Generate 3 augmented variants from a single normalised sequence.
    Returns (fast_warp, slow_warp, jittered).
    """
    T = len(seq)

    # 1. Fast speed warp  (0.75x – 0.90x)
    f1   = np.random.uniform(0.75, 0.90)
    len1 = max(10, int(T * f1))
    idx1 = np.linspace(0, T - 1, len1)
    w1   = np.zeros((len1, seq.shape[1]), dtype=np.float32)
    for c in range(seq.shape[1]):
        w1[:, c] = np.interp(idx1, np.arange(T), seq[:, c])
    w1 = normalize_sequence(resize_sequence(w1, target_length=TARGET_LEN))

    # 2. Slow speed warp  (1.10x – 1.25x)
    f2   = np.random.uniform(1.10, 1.25)
    len2 = max(10, int(T * f2))
    idx2 = np.linspace(0, T - 1, len2)
    w2   = np.zeros((len2, seq.shape[1]), dtype=np.float32)
    for c in range(seq.shape[1]):
        w2[:, c] = np.interp(idx2, np.arange(T), seq[:, c])
    w2 = normalize_sequence(resize_sequence(w2, target_length=TARGET_LEN))

    # 3. Coordinate jitter / sensor noise
    noise    = np.random.normal(0, 0.005, size=seq.shape).astype(np.float32)
    mask     = (seq != 0).astype(np.float32)
    jittered = normalize_sequence((seq + noise * mask).astype(np.float32))

    return w1, w2, jittered


# ============================================================
# PER-CLASS WORKER  (runs in a subprocess via ProcessPoolExecutor)
# ============================================================

def process_single_class(args):
    """
    Returns:
        sign       : class name
        sign_idx   : integer label
        train_seqs : list of np.ndarray (original + augmented) — for training
        val_seqs   : list of np.ndarray (originals only)       — for validation
        skipped    : number of files that could not be processed
    """
    sign, sign_idx, file_paths = args
    rng = np.random.default_rng(RANDOM_SEED + sign_idx)   # deterministic per class

    # ── Source-level 80/20 split ──────────────────────────────────────────────
    n_files   = len(file_paths)
    n_val     = max(1, int(round(n_files * VAL_SPLIT)))
    val_idxs  = set(rng.choice(n_files, size=n_val, replace=False).tolist())
    train_idxs = [i for i in range(n_files) if i not in val_idxs]

    train_seqs = []
    val_seqs   = []
    skipped    = 0

    for i, p_file in enumerate(file_paths):
        if not (p_file.exists() and p_file.stat().st_size > 0):
            skipped += 1
            continue
        try:
            seq = load_landmarks(p_file)
            if len(seq) == 0:
                skipped += 1
                continue
            seq_norm = normalize_sequence(resize_sequence(seq, target_length=TARGET_LEN))
            if seq_norm.shape != (TARGET_LEN, 225) or np.isnan(seq_norm).any():
                skipped += 1
                continue

            if i in val_idxs:
                # ── Validation: ONLY the original clean sequence ──────────────
                val_seqs.append(seq_norm)
            else:
                # ── Training: original + 3 augmented variants ─────────────────
                train_seqs.append(seq_norm)
                w1, w2, jt = augment_sequence(seq_norm)
                if not np.isnan(w1).any():
                    train_seqs.append(w1)
                if not np.isnan(w2).any():
                    train_seqs.append(w2)
                if not np.isnan(jt).any():
                    train_seqs.append(jt)

        except Exception:
            skipped += 1

    # ── Camera-recorded samples (mom & dad) — always go to training ───────────
    cam_dir = DATA_DIR / "camera_mom_dad" / sign
    if cam_dir.exists():
        for cf in cam_dir.glob("*.npy"):
            try:
                cseq = np.load(cf).astype(np.float32)
                cseq_norm = normalize_sequence(cseq)
                if cseq_norm.shape == (TARGET_LEN, 225) and not np.isnan(cseq_norm).any():
                    train_seqs.append(cseq_norm)
                    cw1, cw2, cj = augment_sequence(cseq_norm)
                    if not np.isnan(cw1).any():
                        train_seqs.append(cw1)
                    if not np.isnan(cw2).any():
                        train_seqs.append(cw2)
                    if not np.isnan(cj).any():
                        train_seqs.append(cj)
            except Exception:
                pass

    return sign, sign_idx, train_seqs, val_seqs, skipped


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("SIGNAFLOW AI — BUILDING LEAK-FREE 50-CLASS DATASET (v2)")
    print("Source-level 80/20 split: augmentation only on training files")
    print("=" * 70)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    classes = list(np.load(CLASSES_PATH, allow_pickle=True))
    print(f"Target classes : {len(classes)}")

    df = pd.read_csv(CSV_PATH)
    t0 = time.time()

    # Build task list
    tasks = []
    for sign_idx, sign in enumerate(classes):
        sign_rows  = df[df["sign"] == sign]
        file_paths = [PARQUET_DIR / Path(r["path"]).name for _, r in sign_rows.iterrows()]
        tasks.append((sign, sign_idx, file_paths))

    all_X_train, all_y_train = [], []
    all_X_val,   all_y_val   = [], []

    print("\nProcessing all 50 classes with parallel workers...")
    with ProcessPoolExecutor(max_workers=6) as executor:
        for sign, sign_idx, train_seqs, val_seqs, skipped in executor.map(
            process_single_class, tasks
        ):
            print(
                f"  • {sign:<12} (class {sign_idx:>2}): "
                f"train={len(train_seqs):>5}  val={len(val_seqs):>4}  (skipped={skipped})"
            )
            for s in train_seqs:
                all_X_train.append(s)
                all_y_train.append(sign_idx)
            for s in val_seqs:
                all_X_val.append(s)
                all_y_val.append(sign_idx)

    X_train = np.asarray(all_X_train, dtype=np.float32)
    y_train = np.asarray(all_y_train, dtype=np.int32)
    X_val   = np.asarray(all_X_val,   dtype=np.float32)
    y_val   = np.asarray(all_y_val,   dtype=np.int32)

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("DATASET BUILD COMPLETED!")
    print("=" * 70)
    print(f"  X_train shape : {X_train.shape}")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  X_val   shape : {X_val.shape}")
    print(f"  y_val   shape : {y_val.shape}")
    print(f"  Elapsed       : {elapsed:.1f}s")
    print("=" * 70)

    np.save(X_TRAIN_PATH, X_train)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(X_VAL_PATH,   X_val)
    np.save(Y_VAL_PATH,   y_val)

    print(f"\nSaved:")
    print(f"  {X_TRAIN_PATH}")
    print(f"  {Y_TRAIN_PATH}")
    print(f"  {X_VAL_PATH}")
    print(f"  {Y_VAL_PATH}")
    print("\nNow run:  python src/train_50class_hybrid_v2.py")


if __name__ == "__main__":
    main()
