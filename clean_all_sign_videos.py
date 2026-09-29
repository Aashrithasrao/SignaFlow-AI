from pathlib import Path
import subprocess
import shutil

SIGNS_DIR = Path("signs")
CLEAN_DIR = SIGNS_DIR / "clean"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

videos = [
    p for p in SIGNS_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in VIDEO_EXTENSIONS
    and CLEAN_DIR not in p.parents
]

print("=" * 60)
print("SIGN VIDEO CLEANER")
print("=" * 60)
print(f"Source videos found: {len(videos)}")
print(f"Output directory: {CLEAN_DIR}")
print()

success = 0
failed = 0

for i, src in enumerate(videos, 1):

    # Preserve the sign name from the filename.
    # Example:
    # signs/hello.mp4 -> signs/clean/hello_clean.mp4
    name = src.stem
    output = CLEAN_DIR / f"{name}_clean.mp4"

    print(f"[{i}/{len(videos)}] {src.name}")

    # Skip if already cleaned
    if output.exists() and output.stat().st_size > 0:
        print("   ✓ already exists")
        success += 1
        continue

    # Standardize every video:
    # - H.264 video
    # - AAC audio
    # - 720p maximum
    # - square-ish player-friendly pixel format
    # - fast-start MP4 for Streamlit/browser playback
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),

        "-vf",
        "scale=w=720:h=-2:force_original_aspect_ratio=decrease,"
        "pad=720:720:(ow-iw)/2:(oh-ih)/2:color=black",

        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "22",

        "-c:a", "aac",
        "-b:a", "128k",

        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",

        str(output)
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0 and output.exists():
            print("   ✓ cleaned")
            success += 1
        else:
            print("   ✗ FAILED")
            failed += 1
            print(result.stderr[-500:])

    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        failed += 1

print()
print("=" * 60)
print("FINISHED")
print("=" * 60)
print(f"Successful: {success}")
print(f"Failed:     {failed}")
print(f"Output:     {CLEAN_DIR}")
