import json
import subprocess
import argparse
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WLASL_JSON = (
    PROJECT_ROOT
    / "wlasl_source"
    / "start_kit"
    / "WLASL_v0.3.json"
)

SIGNS_DIR = PROJECT_ROOT / "signs"


# ============================================================
# HELPERS
# ============================================================

def run_ffprobe(video_path):
    """
    Get video FPS and duration using ffprobe.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,duration",
        "-of",
        "default=noprint_wrappers=1",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    values = {}

    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    return values


def parse_fps(value):
    """
    Convert values such as:
        25/1
        30000/1001
        25
    into float FPS.
    """

    if not value:
        return None

    try:
        if "/" in value:
            a, b = value.split("/", 1)
            return float(a) / float(b)

        return float(value)

    except Exception:
        return None


def get_best_instance(entry):
    """
    WLASL downloader normally uses the first available instance.

    Prefer an instance with a real frame range.
    """

    instances = entry.get("instances", [])

    # First preference:
    # actual frame range
    for instance in instances:

        start = instance.get("frame_start")
        end = instance.get("frame_end")

        if start is None or end is None:
            continue

        try:
            start = int(start)
            end = int(end)

            if start >= 1 and end >= start:
                # A real range is better than 1--1
                if not (start == 1 and end == 1):
                    return instance

        except (ValueError, TypeError):
            pass

    # Otherwise use first instance
    if instances:
        return instances[0]

    return None


def crop_video(word, video_path, instance):
    """
    Extract the sign segment using WLASL frame_start/frame_end.
    """

    output_path = video_path.parent / f"{word}_clean.mp4"

    frame_start = instance.get("frame_start")
    frame_end = instance.get("frame_end")
    metadata_fps = instance.get("fps")

    # --------------------------------------------------------
    # If WLASL gives no useful frame range, preserve the video
    # --------------------------------------------------------

    try:
        frame_start = int(frame_start)
        frame_end = int(frame_end)
    except (ValueError, TypeError):
        frame_start = 1
        frame_end = 1

    # --------------------------------------------------------
    # Get actual FPS
    # --------------------------------------------------------

    probe = run_ffprobe(video_path)

    actual_fps = parse_fps(probe.get("r_frame_rate"))

    fps = actual_fps

    if fps is None:
        try:
            fps = float(metadata_fps)
        except (TypeError, ValueError):
            fps = 25.0

    # --------------------------------------------------------
    # No frame range available
    # --------------------------------------------------------

    if frame_start == 1 and frame_end == 1:

        print(f"  No crop range available for {word.upper()}")
        print("  Keeping the downloaded clip as-is.")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

    else:

        # ----------------------------------------------------
        # Frame numbers are 1-based.
        #
        # Therefore:
        # frame 1194 begins at (1194 - 1) / FPS
        # ----------------------------------------------------

        start_time = (frame_start - 1) / fps

        frame_count = frame_end - frame_start + 1

        duration = frame_count / fps

        print(
            f"  Frames      : {frame_start} -> {frame_end}"
        )
        print(
            f"  FPS         : {fps:.3f}"
        )
        print(
            f"  Start       : {start_time:.3f}s"
        )
        print(
            f"  Duration    : {duration:.3f}s"
        )

        # ----------------------------------------------------
        # Preserve original resolution.
        #
        # IMPORTANT:
        # We do NOT upscale because upscaling does not restore
        # detail that was absent from the source.
        # ----------------------------------------------------

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.6f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.6f}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return output_path


# ============================================================
# PROCESS ONE WORD
# ============================================================

def process_word(word, entries_by_word):

    video_path = SIGNS_DIR / word / f"{word}.mp4"

    if not video_path.exists():
        return False

    print()
    print("=" * 70)
    print(f"{word.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # Don't overwrite an existing clean clip
    # --------------------------------------------------------

    clean_path = video_path.parent / f"{word}_clean.mp4"

    if clean_path.exists() and clean_path.stat().st_size > 0:

        print(f"  Already exists:")
        print(f"  {clean_path}")

        return True

    entry = entries_by_word.get(word)

    if entry is None:

        print("  ERROR: WLASL metadata not found.")

        return False

    instance = get_best_instance(entry)

    if instance is None:

        print("  ERROR: No WLASL instances found.")

        return False

    print(f"  Original    : {video_path}")
    print(f"  Source      : {instance.get('source')}")
    print(f"  Video ID    : {instance.get('video_id')}")

    try:

        output = crop_video(
            word,
            video_path,
            instance,
        )

        print()
        print(f"  ✓ Created:")
        print(f"  {output}")

        return True

    except subprocess.CalledProcessError as e:

        print()
        print("  ✗ FFmpeg failed.")
        print(e.stderr[-1000:])

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--word",
        type=str,
        default=None,
        help="Process only one word, e.g. --word good",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Check WLASL metadata
    # --------------------------------------------------------

    if not WLASL_JSON.exists():

        print("ERROR: WLASL metadata not found:")
        print(WLASL_JSON)

        return

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    with open(
        WLASL_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    entries_by_word = {}

    for entry in data:

        word = entry.get("gloss", "").strip().lower()

        if word:
            entries_by_word[word] = entry

    # --------------------------------------------------------
    # Determine words
    # --------------------------------------------------------

    if args.word:

        words = [
            args.word.strip().lower()
        ]

    else:

        words = sorted(
            [
                p.name.lower()
                for p in SIGNS_DIR.iterdir()
                if p.is_dir()
                and (p / f"{p.name}.mp4").exists()
            ]
        )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    success = 0
    failed = 0

    print()
    print("=" * 70)
    print("SIGNAFLOW AI — WLASL VIDEO CLEANER")
    print("=" * 70)

    print(f"Words to process : {len(words)}")

    for word in words:

        result = process_word(
            word,
            entries_by_word,
        )

        if result:
            success += 1
        else:
            failed += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(f"Successful : {success}")
    print(f"Failed     : {failed}")

    print()
    print("Clean videos are saved as:")
    print("signs/<word>/<word>_clean.mp4")


if __name__ == "__main__":
    main()

