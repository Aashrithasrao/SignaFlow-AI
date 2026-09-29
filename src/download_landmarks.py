from pathlib import Path
import pandas as pd
import shutil
import zipfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from kaggle.api.kaggle_api_extended import KaggleApi

CSV_FILE = Path("dataset/train_50_classes.csv")
OUTPUT_DIR = Path("dataset/train_landmarks_50")

MAX_WORKERS = 1
MAX_RETRIES = 8
RETRY_DELAY = 15

_thread_local = threading.local()


def get_api():
    if not hasattr(_thread_local, "api"):
        api = KaggleApi()
        api.authenticate()
        _thread_local.api = api
    return _thread_local.api


def download_one(row):
    remote_path = str(row["path"]).strip()
    filename = Path(remote_path).name
    final_path = OUTPUT_DIR / filename

    if final_path.exists() and final_path.stat().st_size > 0:
        return True, filename, "already exists"

    api = get_api()

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="kaggle_",
            dir=str(OUTPUT_DIR)
        )
    )

    try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Remove anything left from a previous attempt.
                for item in temp_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

                # Kaggle returns:
                # filename.parquet.zip
                api.competition_download_file(
                    competition="asl-signs",
                    file_name=remote_path,
                    path=str(temp_dir),
                    quiet=True,
                )

                # First check for a direct parquet.
                downloaded = temp_dir / filename

                if not downloaded.exists():
                    matches = list(temp_dir.rglob(filename))
                    if matches:
                        downloaded = matches[0]

                # If Kaggle returned a ZIP, extract the parquet.
                if not downloaded.exists():
                    zip_files = list(temp_dir.rglob("*.zip"))

                    for zip_path in zip_files:
                        try:
                            with zipfile.ZipFile(zip_path, "r") as z:
                                members = z.namelist()

                                member = next(
                                    (
                                        name
                                        for name in members
                                        if Path(name).name == filename
                                    ),
                                    None,
                                )

                                if member is None:
                                    continue

                                extract_dir = temp_dir / "extracted"
                                extract_dir.mkdir(
                                    parents=True,
                                    exist_ok=True
                                )

                                z.extract(member, extract_dir)

                                candidate = extract_dir / member

                                if candidate.exists():
                                    downloaded = candidate
                                    break

                        except zipfile.BadZipFile:
                            continue

                if not downloaded.exists():
                    raise FileNotFoundError(
                        f"Could not find {filename} "
                        f"inside Kaggle download"
                    )

                if downloaded.stat().st_size == 0:
                    raise ValueError(
                        f"Downloaded file is empty: {filename}"
                    )

                shutil.move(str(downloaded), str(final_path))

                # Slow down requests to avoid Kaggle HTTP 429 rate limits.
                import time
                time.sleep(2)

                return True, filename, "downloaded"

            except Exception as e:
                if attempt == MAX_RETRIES:
                    return False, filename, str(e)

                import time

                # Exponential backoff, especially important for HTTP 429.
                delay = RETRY_DELAY * (2 ** (attempt - 1))
                print(
                    f"    Retry {attempt}/{MAX_RETRIES} for "
                    f"{filename} in {delay}s..."
                )
                time.sleep(delay)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_FILE)

    # Remove duplicate paths if any.
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)

    total = len(df)

    existing = 0
    missing_rows = []

    for _, row in df.iterrows():
        filename = Path(str(row["path"])).name
        path = OUTPUT_DIR / filename

        if path.exists() and path.stat().st_size > 0:
            existing += 1
        else:
            missing_rows.append(row)

    print("=" * 70)
    print("SIGN LANGUAGE LANDMARK DOWNLOADER")
    print("=" * 70)
    print(f"CSV samples       : {total}")
    print(f"Already downloaded : {existing}")
    print(f"Remaining          : {len(missing_rows)}")
    print(f"Workers            : {MAX_WORKERS}")
    print("=" * 70)

    if not missing_rows:
        print("\nALL LANDMARK FILES ARE ALREADY DOWNLOADED.")
        return

    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_one, row)
            for _, row in pd.DataFrame(missing_rows).iterrows()
        ]

        for future in as_completed(futures):
            completed += 1

            success, filename, message = future.result()

            if success:
                print(
                    f"[{completed}/{len(missing_rows)}] "
                    f"✓ {filename} | {message}"
                )
            else:
                failed += 1
                print(
                    f"[{completed}/{len(missing_rows)}] "
                    f"✗ {filename} | {message}"
                )

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Successfully processed : {completed - failed}")
    print(f"Failed                 : {failed}")

    final_count = len([
        p for p in OUTPUT_DIR.glob("*.parquet")
        if p.is_file() and p.stat().st_size > 0
    ])

    print(f"Total parquet files    : {final_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
