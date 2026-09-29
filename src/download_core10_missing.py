import sys
import tempfile
import zipfile
import shutil
from pathlib import Path
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "dataset" / "train_50_classes.csv"
OUT_DIR = PROJECT_ROOT / "dataset" / "train_landmarks_50"

TARGET_CLASSES = [
    "hello", "mom", "dad", "water", "food",
    "yes", "no", "please", "go", "home"
]


def main():
    print("=" * 70)
    print("DOWNLOADING ALL REMAINING KAGGLE PARQUETS FOR 10 CORE WORDS")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)
    sub_df = df[df["sign"].isin(TARGET_CLASSES)]

    missing = []
    for _, row in sub_df.iterrows():
        p_name = Path(row["path"]).name
        if not (OUT_DIR / p_name).exists():
            missing.append(row)

    print(f"Total files to download: {len(missing)}")
    if not missing:
        print("All files are already downloaded!")
        return

    api = KaggleApi()
    api.authenticate()

    def dl(row):
        remote = str(row["path"]).strip()
        fname = Path(remote).name
        temp_dir = Path(tempfile.mkdtemp(prefix="kaggle_core_"))
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
            return False, f"{fname}: {e}"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return False, fname

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(dl, missing))

    successes = sum(1 for s, _ in results if s)
    print(f"\nDownload completed: {successes}/{len(missing)} files saved to {OUT_DIR.name}")


if __name__ == "__main__":
    main()
