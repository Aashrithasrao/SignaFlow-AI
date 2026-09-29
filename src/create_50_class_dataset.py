import pandas as pd

INPUT_CSV = "dataset/train.csv"
OUTPUT_CSV = "dataset/train_50_classes.csv"

SELECTED_CLASSES = [
    "hello",
    "thankyou",
    "please",
    "yes",
    "no",
    "better",
    "bad",
    "happy",
    "sad",
    "morning",
    "night",
    "home",
    "water",
    "food",
    "drink",
    "hungry",
    "thirsty",
    "listen",
    "hear",
    "see",
    "go",
    "look",
    "give",
    "open",
    "close",
    "wait",
    "sleep",
    "wake",
    "read",
    "book",
    "boy",
    "girl",
    "man",
    "mom",
    "dad",
    "later",
    "now",
    "where",
    "who",
    "why",
    "have",
    "fine",
    "every",
    "first",
    "say",
    "talk",
    "same",
    "stay",
    "outside",
    "tomorrow",
]

# Load original dataset
df = pd.read_csv(INPUT_CSV)

# Check that every requested class exists
available_classes = set(df["sign"].unique())

missing_classes = [
    sign for sign in SELECTED_CLASSES
    if sign not in available_classes
]

if missing_classes:
    print("ERROR: These classes were not found:")
    for sign in missing_classes:
        print(" -", sign)
    raise SystemExit(1)

# Select only our 50 classes
subset = df[df["sign"].isin(SELECTED_CLASSES)].copy()

# Save
subset.to_csv(OUTPUT_CSV, index=False)

print("=" * 60)
print("50-CLASS DATASET CREATED")
print("=" * 60)

print(f"Total samples : {len(subset):,}")
print(f"Total classes : {subset['sign'].nunique()}")

print("\nSamples per class:")
print(subset["sign"].value_counts().sort_index())

print("\nSaved to:")
print(OUTPUT_CSV)

print("=" * 60)
