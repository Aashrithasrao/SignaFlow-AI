import pandas as pd

FILE = "dataset/sample_landmarks/1000035562.parquet"

print("=" * 70)
print("LANDMARK FILE INSPECTION")
print("=" * 70)

df = pd.read_parquet(FILE)

print("\n📊 BASIC INFORMATION")
print("-" * 70)
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n📋 COLUMN NAMES")
print("-" * 70)

for i, column in enumerate(df.columns):
    print(f"{i:3}: {column}")

print("\n🔍 DATA TYPES")
print("-" * 70)
print(df.dtypes.to_string())

print("\n👀 FIRST 5 ROWS")
print("-" * 70)
print(df.head().to_string())

print("\n❓ MISSING VALUES")
print("-" * 70)

missing = df.isna().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing.to_string())

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
