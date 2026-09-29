import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from src.preprocess_landmarks import (
    load_landmarks,
    resize_sequence,
    normalize_sequence,
)


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = Path(
    "dataset/train_50_classes.csv"
)

LANDMARK_DIR = Path(
    "dataset/train_landmarks_50"
)

OUTPUT_DIR = Path(
    "dataset/processed"
)

SEQUENCE_LENGTH = 32
FEATURES = 225


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SIGN LANGUAGE — FULL TRAINING DATA BUILDER")
print("=" * 70)

print(
    f"\nCSV:"
    f" {CSV_PATH}"
)

print(
    f"Landmarks:"
    f" {LANDMARK_DIR}"
)

print(
    f"Output:"
    f" {OUTPUT_DIR}"
)


# ============================================================
# LOAD CSV
# ============================================================

if not CSV_PATH.exists():

    raise FileNotFoundError(
        f"CSV not found: {CSV_PATH}"
    )


df = pd.read_csv(
    CSV_PATH
)

print(
    f"\nTotal labeled samples in CSV: "
    f"{len(df):,}"
)


# ============================================================
# CREATE CLASS MAPPING
# ============================================================

classes = sorted(
    df["sign"]
    .dropna()
    .unique()
)

label_map = {
    sign: index
    for index, sign in enumerate(classes)
}


print(
    f"Total classes in CSV: "
    f"{len(classes)}"
)


print("\nClass mapping:")

for sign, index in label_map.items():

    print(
        f"{index:2d} → {sign}"
    )


# ============================================================
# FIND DOWNLOADED FILES
# ============================================================

downloaded_files = {
    p.name
    for p in LANDMARK_DIR.glob(
        "*.parquet"
    )
}


print(
    f"\nDownloaded parquet files: "
    f"{len(downloaded_files):,}"
)


# ============================================================
# MATCH DOWNLOADED FILES TO CSV
# ============================================================

available_rows = []

for _, row in df.iterrows():

    filename = Path(
        row["path"]
    ).name

    if filename in downloaded_files:

        available_rows.append(
            row
        )


print(
    f"Labeled downloaded samples: "
    f"{len(available_rows):,}"
)


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

available_class_counts = Counter()

for row in available_rows:

    available_class_counts[
        row["sign"]
    ] += 1


print("\nDownloaded samples by class:")
print("-" * 45)

for sign in classes:

    count = available_class_counts.get(
        sign,
        0
    )

    print(
        f"{sign:20s} : {count:4d}"
    )


missing_classes = [
    sign
    for sign in classes
    if available_class_counts.get(
        sign,
        0
    ) == 0
]


if missing_classes:

    print(
        "\nWARNING:"
    )

    print(
        "These classes have NO downloaded samples:"
    )

    for sign in missing_classes:

        print(
            f"  - {sign}"
        )


# ============================================================
# PROCESS LANDMARKS
# ============================================================

X = []
y = []

processed = 0
skipped = 0

processed_class_counts = Counter()


print("\n" + "=" * 70)
print("PROCESSING LANDMARK FILES")
print("=" * 70)


for row in available_rows:

    filename = Path(
        row["path"]
    ).name

    parquet_path = (
        LANDMARK_DIR /
        filename
    )

    sign = row["sign"]

    try:

        # ----------------------------------------------------
        # LOAD RAW LANDMARKS
        # ----------------------------------------------------

        sequence = load_landmarks(
            parquet_path
        )

        # ----------------------------------------------------
        # RESIZE TO 32 FRAMES
        # ----------------------------------------------------

        sequence = resize_sequence(
            sequence,
            SEQUENCE_LENGTH
        )

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        sequence = normalize_sequence(
            sequence
        )

        # ----------------------------------------------------
        # VALIDATE SHAPE
        # ----------------------------------------------------

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURES
        ):

            print(
                f"\nSKIP {filename}"
            )

            print(
                f"Unexpected shape: "
                f"{sequence.shape}"
            )

            skipped += 1

            continue

        # ----------------------------------------------------
        # VALIDATE NUMBERS
        # ----------------------------------------------------

        if np.isnan(
            sequence
        ).any():

            print(
                f"\nSKIP {filename}: "
                f"NaN detected"
            )

            skipped += 1

            continue

        if np.isinf(
            sequence
        ).any():

            print(
                f"\nSKIP {filename}: "
                f"Infinite value detected"
            )

            skipped += 1

            continue

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        X.append(
            sequence
        )

        y.append(
            label_map[sign]
        )

        processed += 1

        processed_class_counts[
            sign
        ] += 1

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if processed % 100 == 0:

            print(
                f"Processed: "
                f"{processed:,} / "
                f"{len(available_rows):,}"
            )

    except Exception as e:

        print(
            f"\nERROR:"
            f" {filename}"
        )

        print(
            f"Reason:"
            f" {e}"
        )

        skipped += 1


# ============================================================
# CONVERT TO NUMPY
# ============================================================

print(
    "\nConverting to NumPy..."
)

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int64
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET VALIDATION")
print("=" * 70)

print(
    f"\nProcessed samples:"
    f" {processed:,}"
)

print(
    f"Skipped samples:"
    f" {skipped:,}"
)

print(
    f"\nX shape:"
    f" {X.shape}"
)

print(
    f"y shape:"
    f" {y.shape}"
)

print(
    f"\nX dtype:"
    f" {X.dtype}"
)

print(
    f"y dtype:"
    f" {y.dtype}"
)


# ============================================================
# SAFETY CHECK
# ============================================================

if len(X) == 0:

    raise RuntimeError(
        "No training samples were created."
    )


if X.shape[1:] != (
    SEQUENCE_LENGTH,
    FEATURES
):

    raise RuntimeError(
        f"Unexpected X shape: {X.shape}"
    )


if len(X) != len(y):

    raise RuntimeError(
        "X and y have different lengths."
    )


# ============================================================
# NUMERICAL CHECKS
# ============================================================

nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()

print(
    f"\nNaN values:"
    f" {nan_count:,}"
)

print(
    f"Infinite values:"
    f" {inf_count:,}"
)

print(
    f"\nMinimum:"
    f" {X.min():.6f}"
)

print(
    f"Maximum:"
    f" {X.max():.6f}"
)


# ============================================================
# FINAL CLASS DISTRIBUTION
# ============================================================

print(
    "\nFinal processed class distribution:"
)

print("-" * 45)

for sign in classes:

    count = processed_class_counts.get(
        sign,
        0
    )

    print(
        f"{sign:20s} : {count:4d}"
    )


# ============================================================
# SAVE DATA
# ============================================================

print(
    "\nSaving processed dataset..."
)

np.save(
    OUTPUT_DIR / "X.npy",
    X
)

np.save(
    OUTPUT_DIR / "y.npy",
    y
)

np.save(
    OUTPUT_DIR / "classes.npy",
    np.asarray(
        classes
    )
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("DATASET BUILD COMPLETE")
print("=" * 70)

print(
    "\nSaved:"
)

print(
    f"  {OUTPUT_DIR / 'X.npy'}"
)

print(
    f"  {OUTPUT_DIR / 'y.npy'}"
)

print(
    f"  {OUTPUT_DIR / 'classes.npy'}"
)

print(
    "\nFinal dataset:"
)

print(
    f"  Samples : {len(X):,}"
)

print(
    f"  Classes : {len(classes)}"
)

print(
    f"  Shape   : {X.shape}"
)

print(
    "\n" + "=" * 70
)
