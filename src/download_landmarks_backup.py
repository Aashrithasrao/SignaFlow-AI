import pandas as pd
import subprocess
from pathlib import Path
import zipfile


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CSV_PATH = "dataset/train_50_classes.csv"

OUTPUT_DIR = Path("dataset/train_landmarks_50")

SAMPLES_PER_CLASS = 50


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(CSV_PATH)

# Exactly 50 samples from every class
samples = (
    df.groupby("sign", group_keys=False)
      .head(SAMPLES_PER_CLASS)
      .copy()
)

print("=" * 60)
print("50-SAMPLE-PER-CLASS LANDMARK DOWNLOAD")
print("=" * 60)

print(f"Classes: {samples['sign'].nunique()}")
print(f"Target samples: {len(samples)}")
print(f"Samples per class: {SAMPLES_PER_CLASS}")

successful = 0
downloaded_now = 0
failed = 0


# ---------------------------------------------------------
# Download
# ---------------------------------------------------------

for count, (_, row) in enumerate(
    samples.iterrows(),
    start=1
):

    remote_path = row["path"]

    filename = Path(remote_path).name

    parquet_path = OUTPUT_DIR / filename
    zip_path = OUTPUT_DIR / f"{filename}.zip"

    # Already extracted
    if parquet_path.exists():

        successful += 1

        print(
            f"[{count}/{len(samples)}] "
            f"EXISTS: {filename}"
        )

        continue

    # ZIP exists → extract it
    if zip_path.exists():

        try:

            with zipfile.ZipFile(
                zip_path,
                "r"
            ) as zip_ref:

                zip_ref.extractall(
                    OUTPUT_DIR
                )

            zip_path.unlink()

            successful += 1

            print(
                f"[{count}/{len(samples)}] "
                f"EXTRACTED: {filename}"
            )

        except Exception as e:

            failed += 1

            print(
                f"[{count}/{len(samples)}] "
                f"EXTRACTION FAILED: {e}"
            )

        continue


    # Download
    print(
        f"[{count}/{len(samples)}] "
        f"DOWNLOADING: {row['sign']}"
    )

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

        failed += 1

        print(
            f"  FAILED: {remote_path}"
        )

        continue


    # Extract downloaded ZIP
    if zip_path.exists():

        try:

            with zipfile.ZipFile(
                zip_path,
                "r"
            ) as zip_ref:

                zip_ref.extractall(
                    OUTPUT_DIR
                )

            zip_path.unlink()

        except Exception as e:

            failed += 1

            print(
                f"  EXTRACTION FAILED: {e}"
            )

            continue


    if parquet_path.exists():

        successful += 1
        downloaded_now += 1

    else:

        failed += 1

        print(
            f"  FILE NOT FOUND AFTER DOWNLOAD: "
            f"{filename}"
        )


# ---------------------------------------------------------
# Final verification
# ---------------------------------------------------------

parquet_files = list(
    OUTPUT_DIR.glob("*.parquet")
)

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(
    f"Target samples : {len(samples)}"
)

print(
    f"Parquet files  : {len(parquet_files)}"
)

print(
    f"Already/existing: "
    f"{successful - downloaded_now}"
)

print(
    f"Downloaded now : {downloaded_now}"
)

print(
    f"Failed         : {failed}"
)

print(
    f"Directory      : {OUTPUT_DIR}"
)

print("=" * 60)
