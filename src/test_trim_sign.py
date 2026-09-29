import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT = PROJECT_ROOT / "signs" / "good" / "good.mp4"
OUTPUT = PROJECT_ROOT / "signs" / "good" / "good_trimmed.mp4"

FPS = 25
FRAME_START = 1194
FRAME_END = 1236

start_time = FRAME_START / FPS
duration = (FRAME_END - FRAME_START + 1) / FPS

print(f"Input    : {INPUT}")
print(f"Start    : {start_time:.3f} sec")
print(f"Duration : {duration:.3f} sec")
print(f"Output   : {OUTPUT}")

command = [
    "ffmpeg",
    "-y",
    "-ss",
    str(start_time),
    "-i",
    str(INPUT),
    "-t",
    str(duration),
    "-an",
    "-vf",
    "scale=1280:-2:flags=lanczos",
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-crf",
    "20",
    "-pix_fmt",
    "yuv420p",
    str(OUTPUT),
]

subprocess.run(command, check=True)

print()
print("DONE")
print(OUTPUT)
