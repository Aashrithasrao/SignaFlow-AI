import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WLASL_JSON = PROJECT_ROOT / "wlasl_source" / "start_kit" / "WLASL_v0.3.json"
SIGNS_DIR = PROJECT_ROOT / "signs"


def main():
    if not WLASL_JSON.exists():
        print("ERROR: WLASL metadata not found:")
        print(WLASL_JSON)
        return

    with open(WLASL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 80)
    print("WLASL DOWNLOADED ASSET CHECK")
    print("=" * 80)

    total = 0
    found = 0

    for entry in data:
        word = entry.get("gloss", "").strip().lower()

        video_path = SIGNS_DIR / word / f"{word}.mp4"

        if not video_path.exists():
            continue

        found += 1
        total += 1

        instances = entry.get("instances", [])

        print()
        print(f"[{found}] {word.upper()}")
        print(f"    Local : {video_path}")

        if not instances:
            print("    WLASL : NO INSTANCES")
            continue

        print(f"    WLASL instances: {len(instances)}")

        # Show first 5 instances
        for i, inst in enumerate(instances[:5]):
            print(
                f"    #{i + 1}: "
                f"video_id={inst.get('video_id')} | "
                f"source={inst.get('source')} | "
                f"fps={inst.get('fps')} | "
                f"frames={inst.get('frame_start')}-{inst.get('frame_end')} | "
                f"url={inst.get('url')}"
            )

    print()
    print("=" * 80)
    print(f"Downloaded WLASL word videos found: {total}")
    print("=" * 80)


if __name__ == "__main__":
    main()
