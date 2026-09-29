import pandas as pd
from pathlib import Path


CSV_PATH = Path("dataset/train_50_classes.csv")
LANDMARK_DIR = Path("dataset/train_landmarks_50")

OUTPUT = Path(
    "dataset/remaining_landmarks.txt"
)


df = pd.read_csv(CSV_PATH)

downloaded = {
    p.name
    for p in LANDMARK_DIR.glob("*.parquet")
}


remaining = []

for _, row in df.iterrows():

    filename = Path(
        row["path"]
    ).name

    if filename not in downloaded:

        remaining.append(
            row["path"]
        )


with open(
    OUTPUT,
    "w"
) as f:

    for path in remaining:

        f.write(
            path + "\n"
        )


print("=" * 60)
print("DOWNLOAD LIST CREATED")
print("=" * 60)

print(
    f"Total labeled samples : {len(df):,}"
)

print(
    f"Already downloaded    : {len(downloaded):,}"
)

print(
    f"Remaining              : {len(remaining):,}"
)

print(
    f"\nSaved to:"
)

print(
    OUTPUT
)
