import pandas as pd
from pathlib import Path

INPUT = Path("dataset/train_50_classes.csv")
OUTPUT = Path("dataset/train_5_classes.csv")

CLASSES = ["hello", "mom", "dad", "water", "food"]

df = pd.read_csv(INPUT)

df5 = df[df["sign"].str.lower().isin(CLASSES)].copy()

# Keep only files that actually exist locally
available = {
    p.stem
    for p in Path("dataset/train_landmarks_50").glob("*.parquet")
}

df5["file_id"] = (
    df5["path"]
    .astype(str)
    .str.rsplit("/", n=1)
    .str[-1]
    .str.replace(".parquet", "", regex=False)
)

df5 = df5[df5["file_id"].isin(available)].copy()

# Fixed class order
mapping = {
    "hello": 0,
    "mom": 1,
    "dad": 2,
    "water": 3,
    "food": 4,
}

df5["label"] = df5["sign"].str.lower().map(mapping)

df5.to_csv(OUTPUT, index=False)

print("=" * 60)
print("5-CLASS DATASET")
print("=" * 60)
print("Classes:", CLASSES)
print("Total samples:", len(df5))
print()

for name in CLASSES:
    print(f"{name:10} : {(df5['sign'].str.lower() == name).sum()}")

print()
print("Saved:", OUTPUT)
