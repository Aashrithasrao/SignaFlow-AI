"""
build_50class_clean.py
======================
Builds a CLEAN, NO-AUGMENTATION 50-class dataset.

- Loads original Kaggle parquet landmark files only.
- Normalises + resizes each sequence to (32, 225).
- Splits source files 80% train / 20% val at the FILE level (no leakage).
- NO speed warping, NO jitter, NO synthetic augmentation.

This gives an honest baseline: what can the model learn from
real human signing data alone?

Output (overwrites pre-split files used by train_50class_hybrid_v2.py):
  dataset/processed/X_train.npy
  dataset/processed/y_train.npy
  dataset/processed/X_val.npy
  dataset/processed/y_val.npy
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess_landmarks import load_landmarks, resize_sequence, normalize_sequence

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR     = PROJECT_ROOT / "dataset"
PARQUET_DIR  = DATA_DIR / "train_landmarks_50"
CSV_PATH     = DATA_DIR / "train_50_classes.csv"
PROC_DIR     = DATA_DIR / "processed"
CLASSES_PATH = PROC_DIR / "classes.npy"

X_TRAIN_PATH = PROC_DIR / "X_train.npy"
Y_TRAIN_PATH = PROC_DIR / "y_train.npy"
X_VAL_PATH   = PROC_DIR / "X_val.npy"
Y_VAL_PATH   = PROC_DIR / "y_val.npy"

TARGET_LEN  = 32
VAL_SPLIT   = 0.20
RANDOM_SEED = 42


# ── Per-class worker ─────────────────────────────────────────────────────────

def process_single_class(args):
    sign, sign_idx, file_paths = args
    rng = np.random.default_rng(RANDOM_SEED + sign_idx)

    n_files   = len(file_paths)
    n_val     = max(1, int(round(n_files * VAL_SPLIT)))
    val_idxs  = set(rng.choice(n_files, size=n_val, replace=False).tolist())

    train_seqs, val_seqs = [], []
    skipped = 0

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
                val_seqs.append(seq_norm)
            else:
                train_seqs.append(seq_norm)

        except Exception:
            skipped += 1

    # Camera samples always go to training (no augmentation applied)
    cam_dir = DATA_DIR / "camera_mom_dad" / sign
    if cam_dir.exists():
        for cf in cam_dir.glob("*.npy"):
            try:
                cseq = np.load(cf).astype(np.float32)
                cseq_norm = normalize_sequence(cseq)
                if cseq_norm.shape == (TARGET_LEN, 225) and not np.isnan(cseq_norm).any():
                    train_seqs.append(cseq_norm)
            except Exception:
                pass

    return sign, sign_idx, train_seqs, val_seqs, skipped


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SIGNAFLOW AI — CLEAN 50-CLASS DATASET (No Augmentation)")
    print("Source-level 80/20 split — original data only")
    print("=" * 70)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    classes = list(np.load(CLASSES_PATH, allow_pickle=True))
    print(f"Classes : {len(classes)}")

    df = pd.read_csv(CSV_PATH)
    t0 = time.time()

    tasks = []
    for sign_idx, sign in enumerate(classes):
        rows       = df[df["sign"] == sign]
        file_paths = [PARQUET_DIR / Path(r["path"]).name for _, r in rows.iterrows()]
        tasks.append((sign, sign_idx, file_paths))

    all_X_train, all_y_train = [], []
    all_X_val,   all_y_val   = [], []

    print("\nProcessing all 50 classes...")
    with ProcessPoolExecutor(max_workers=6) as executor:
        for sign, sign_idx, train_seqs, val_seqs, skipped in executor.map(
            process_single_class, tasks
        ):
            print(
                f"  • {sign:<12} (class {sign_idx:>2}): "
                f"train={len(train_seqs):>4}  val={len(val_seqs):>3}  (skipped={skipped})"
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

    print("\n" + "=" * 70)
    print("DONE!")
    print(f"  X_train : {X_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  y_val   : {y_val.shape}")
    print(f"  Time    : {time.time()-t0:.1f}s")
    print("=" * 70)

    np.save(X_TRAIN_PATH, X_train)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(X_VAL_PATH,   X_val)
    np.save(Y_VAL_PATH,   y_val)

    print(f"\nSaved to {PROC_DIR}/")
    print("Now run:  ./.venv/bin/python src/train_50class_hybrid_v2.py")


if __name__ == "__main__":
    main()
