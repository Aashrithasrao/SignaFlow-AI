import sys
import tempfile
import zipfile
import shutil
import time
from pathlib import Path
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "dataset" / "train_50_classes.csv"
OUT_DIR = PROJECT_ROOT / "dataset" / "train_landmarks_50"

MAX_WORKERS = 8


def main():
    print("=" * 70)
    print("SIGNAFLOW AI — DOWNLOADING ALL 796 REMAINING KAGGLE 50-CLASS FILES")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    missing = []
    for _, row in df.iterrows():
        p_name = Path(row["path"]).name
        if not (OUT_DIR / p_name).exists():
            missing.append(row)

    total_missing = len(missing)
    print(f"Total CSV entries   : {len(df):,}")
    print(f"Already downloaded  : {len(df) - total_missing:,}")
    print(f"Remaining to download: {total_missing:,}")
    print("=" * 70)

    if total_missing == 0:
        print("All 18,948 files are already downloaded!")
        return

    api = KaggleApi()
    api.authenticate()

    def dl_file(row):
        remote = str(row["path"]).strip()
        fname = Path(remote).name
        
        for attempt in range(3):
            temp_dir = Path(tempfile.mkdtemp(prefix="kaggle_50_"))
            try:
                api.competition_download_file(
                    competition="asl-signs",
                    file_name=remote,
                    path=str(temp_dir),
                    quiet=True
                )
                for z in temp_dir.rglob("*.zip"):
                    with zipfile.ZipFile(z, "r") as ref:
                        ref.extractall(temp_dir)
                cands = list(temp_dir.rglob(fname))
                if cands:
                    shutil.move(str(cands[0]), str(OUT_DIR / fname))
                    return True, fname
            except Exception as e:
                if attempt == 2:
                    return False, f"{fname}: {e}"
                time.sleep(1.5 * (attempt + 1))
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        return False, fname

    completed = 0
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(dl_file, row): row for row in missing}
        for future in as_completed(futures):
            completed += 1
            success, name = future.result()
            if not success:
                failed += 1

            if completed % 50 == 0 or completed == total_missing:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"Progress: [{completed}/{total_missing}] files ({completed/total_missing*100:.1f}%) | Speed: {rate:.1f} files/sec | Failed: {failed}")

    print("\n" + "=" * 70)
    print(f"DOWNLOAD COMPLETE: {completed - failed}/{total_missing} downloaded successfully in {time.time() - start_time:.1f}s")
    print(f"Total parquet files now in {OUT_DIR.name}: {len(list(OUT_DIR.glob('*.parquet')))}")
    print("=" * 70)


if __name__ == "__main__":
    main()
