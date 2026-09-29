import json
import subprocess
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WLASL_JSON = (
    PROJECT_ROOT
    / "wlasl_source"
    / "start_kit"
    / "WLASL_v0.3.json"
)

OUTPUT_DIR = PROJECT_ROOT / "signs"

MAX_WORKERS = 4
MAX_RETRIES_PER_VIDEO = 2
VIDEO_TIMEOUT = 120
MIN_FILE_SIZE = 10_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEFAULT SIGN VOCABULARY
# ============================================================

REQUIRED_WORDS = """
hello good morning afternoon night
i you me my name
go come college school home
water food eat drink
please yes no
happy sad want need have
give take help
where what who why how when
today tomorrow yesterday now later
listen see hear read write
talk say know understand like love
work study friend family
mother father boy girl man woman
book open close sleep wait
same better fine outside first every
""".split()


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="SignalFlow AI WLASL video downloader"
)

parser.add_argument(
    "--words-file",
    type=str,
    default=None,
    help="Text file containing one sign word per line",
)

args = parser.parse_args()


# ============================================================
# LOAD REQUESTED VOCABULARY
# ============================================================

if args.words_file:
    words_path = Path(args.words_file)

    if not words_path.exists():
        print(f"ERROR: Words file not found: {words_path}")
        raise SystemExit(1)

    REQUIRED_WORDS = [
        line.strip().lower()
        for line in words_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Remove duplicates while preserving order.
    REQUIRED_WORDS = list(dict.fromkeys(REQUIRED_WORDS))


# ============================================================
# LOAD WLASL
# ============================================================

print()
print("=" * 70)
print("SIGNALFLOW AI — PARALLEL WLASL VIDEO DOWNLOADER")
print("=" * 70)
print()

if not WLASL_JSON.exists():
    print("ERROR: WLASL metadata not found:")
    print(WLASL_JSON)
    raise SystemExit(1)

with open(WLASL_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)


# ============================================================
# INDEX WLASL CLASSES
# ============================================================

wlasl = {}

for item in data:
    gloss = item["gloss"].strip().lower()
    wlasl[gloss] = item["instances"]


available_words = [
    word for word in REQUIRED_WORDS
    if word in wlasl
]

missing_words = [
    word for word in REQUIRED_WORDS
    if word not in wlasl
]


print(f"Requested vocabulary : {len(REQUIRED_WORDS)}")
print(f"Available in WLASL  : {len(available_words)}")
print(f"Missing from WLASL  : {len(missing_words)}")

if missing_words:
    print()
    print("Missing:")
    print(", ".join(missing_words))

print()


# ============================================================
# URL CLEANING
# ============================================================

def clean_url(url):
    if not url:
        return ""

    url = str(url).strip()

    url = url.replace("[", "").replace("]", "")

    if "|" in url:
        url = url.split("|")[-1]

    return url.strip()


# ============================================================
# DOWNLOAD ONE WORD
# ============================================================

def download_word(word):

    word_dir = OUTPUT_DIR / word
    word_dir.mkdir(parents=True, exist_ok=True)

    final_file = word_dir / f"{word}.mp4"

    if (
        final_file.exists()
        and final_file.stat().st_size >= MIN_FILE_SIZE
    ):
        return {
            "word": word,
            "status": "exists",
            "file": str(final_file),
        }

    instances = wlasl[word]

    instances = sorted(
        instances,
        key=lambda x: (
            0 if x.get("split") == "train" else 1,
            x.get("variation_id", 0),
        ),
    )

    for instance in instances:

        video_id = str(instance.get("video_id", "unknown"))
        source = str(instance.get("source", "unknown"))
        url = clean_url(instance.get("url", ""))

        if not url:
            continue

        temp_file = word_dir / f".tmp_{video_id}.mp4"

        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            pass

        command = [
            "yt-dlp",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--retries",
            str(MAX_RETRIES_PER_VIDEO),
            "--fragment-retries",
            str(MAX_RETRIES_PER_VIDEO),
            "--socket-timeout",
            "30",
            "-f",
            "mp4/best",
            "-o",
            str(temp_file),
            url,
        ]

        try:

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=VIDEO_TIMEOUT,
            )

            if (
                result.returncode == 0
                and temp_file.exists()
                and temp_file.stat().st_size >= MIN_FILE_SIZE
            ):

                temp_file.replace(final_file)

                return {
                    "word": word,
                    "status": "downloaded",
                    "file": str(final_file),
                    "video_id": video_id,
                    "source": source,
                }

        except subprocess.TimeoutExpired:
            pass

        except Exception:
            pass

        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            pass

    return {
        "word": word,
        "status": "failed",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STARTING PARALLEL DOWNLOAD")
    print("=" * 70)
    print()

    print(f"Workers : {MAX_WORKERS}")
    print(f"Words   : {len(available_words)}")
    print()

    start_time = time.time()

    downloaded = 0
    already_exists = 0
    failed = 0

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_map = {
            executor.submit(download_word, word): word
            for word in available_words
        }

        total = len(future_map)

        for number, future in enumerate(
            as_completed(future_map),
            start=1,
        ):

            word = future_map[future]

            try:
                result = future.result()

            except Exception as exc:

                failed += 1

                print(
                    f"[{number}/{total}] "
                    f"✗ {word:<14} "
                    f"ERROR: {exc}",
                    flush=True,
                )

                continue

            status = result["status"]

            if status == "downloaded":

                downloaded += 1

                print(
                    f"[{number}/{total}] "
                    f"✓ {word:<14} "
                    f"DOWNLOADED",
                    flush=True,
                )

            elif status == "exists":

                already_exists += 1

                print(
                    f"[{number}/{total}] "
                    f"✓ {word:<14} "
                    f"ALREADY EXISTS",
                    flush=True,
                )

            else:

                failed += 1

                print(
                    f"[{number}/{total}] "
                    f"✗ {word:<14} "
                    f"FAILED",
                    flush=True,
                )

    elapsed = (time.time() - start_time) / 60

    video_files = list(
        OUTPUT_DIR.glob("*/*.mp4")
    )

    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print()

    print(f"Downloaded this run : {downloaded}")
    print(f"Already existed     : {already_exists}")
    print(f"Failed              : {failed}")
    print(f"Total MP4 files     : {len(video_files)}")
    print(f"Time                : {elapsed:.1f} minutes")

    print()

    if failed:
        print(
            "Some videos failed. "
            "Run the same command again to retry them."
        )

    print()

    print("Missing from WLASL:")
    print(", ".join(missing_words))

    print()

    print("=" * 70)
    print("VIDEO DOWNLOAD FINISHED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
