import pandas as pd
import json

CSV_PATH = "dataset/train.csv"
JSON_PATH = "dataset/sign_to_prediction_index_map.json"

print("=" * 60)
print("SIGN LANGUAGE DATASET INSPECTION")
print("=" * 60)

# Load training CSV
df = pd.read_csv(CSV_PATH)

print("\n📊 BASIC INFORMATION")
print("-" * 60)
print(f"Total samples       : {len(df):,}")
print(f"Number of classes   : {df['sign'].nunique()}")
print(f"Participants        : {df['participant_id'].nunique()}")
print(f"Columns             : {list(df.columns)}")

print("\n📝 FIRST 10 SAMPLES")
print("-" * 60)
print(df.head(10).to_string(index=False))

print("\n🏷️ CLASS DISTRIBUTION")
print("-" * 60)

class_counts = df["sign"].value_counts()

print(f"Minimum samples/class: {class_counts.min()}")
print(f"Maximum samples/class: {class_counts.max()}")
print(f"Average samples/class: {class_counts.mean():.2f}")

print("\nMost common classes:")
print(class_counts.head(15).to_string())

print("\nLeast common classes:")
print(class_counts.tail(15).to_string())

print("\n👥 PARTICIPANT DISTRIBUTION")
print("-" * 60)
print(df["participant_id"].value_counts().sort_index().to_string())

# Load sign mapping
with open(JSON_PATH, "r") as f:
    sign_map = json.load(f)

print("\n🔤 SIGN MAPPING")
print("-" * 60)
print(f"Number of mapped signs: {len(sign_map)}")

print("\nFirst 20 signs:")
for sign, index in list(sign_map.items())[:20]:
    print(f"{index:3} → {sign}")

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)
