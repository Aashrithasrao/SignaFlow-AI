import pandas as pd
import subprocess
from pathlib import Path

CSV_PATH = "dataset/train_50_classes.csv"
OUTPUT_DIR = Path("dataset/sequence_samples")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Select one sample from every class first
samples = (
    df.groupby("sign", group_keys=False)
      .head(1)
)

# Take 20 classes for our initial inspection
samples = samples.head(20)

print("=" * 60)
print("DOWNLOADING REPRESENTATIVE LANDMARK FILES")
print("=" * 60)

for _, row in samples.iterrows():

    remote_path = row["path"]

    filename = Path(remote_path).name

    output_file = OUTPUT_DIR / filename

    print(f"\nDownloading: {row['sign']}")
    print(f"File: {remote_path}")

    if output_file.exists():
        print("Already downloaded.")
        continue

    command = [
        "kaggle",
        "competitions",
        "download",
        "asl-signs",
        "-f",
        remote_path,
        "-p",
        str(OUTPUT_DIR),
    ]

    result = subprocess.run(command)

    if result.returncode != 0:
        print("ERROR downloading:", remote_path)

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)
