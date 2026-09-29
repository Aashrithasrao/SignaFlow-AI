import pandas as pd
from pathlib import Path

DATA_DIR = Path("dataset/sequence_samples")

results = []

for file in DATA_DIR.glob("*.parquet"):
    df = pd.read_parquet(file)

    results.append({
        "file": file.name,
        "frames": df["frame"].nunique()
    })

stats = pd.DataFrame(results)

print("=" * 60)
print("SEQUENCE LENGTH ANALYSIS")
print("=" * 60)

print("\nIndividual samples:")
print(stats.to_string(index=False))

print("\n📊 STATISTICS")
print("-" * 60)

print(f"Samples analyzed : {len(stats)}")
print(f"Minimum frames   : {stats['frames'].min()}")
print(f"Maximum frames   : {stats['frames'].max()}")
print(f"Mean frames      : {stats['frames'].mean():.2f}")
print(f"Median frames    : {stats['frames'].median():.2f}")
print(f"95th percentile  : {stats['frames'].quantile(0.95):.2f}")

print("\n" + "=" * 60)
