import cv2
import numpy as np
from pathlib import Path
import sys

from camera_landmarks import CameraLandmarkExtractor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "signs" / "water" / "water_clean.mp4"
OUTPUT_DIR = PROJECT_ROOT / "sign_motion"


def extract_video(video_path: Path):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 70)
    print("SIGNAFLOW — SIGN MOTION EXTRACTION")
    print("=" * 70)
    print(f"Video  : {video_path}")
    print(f"FPS    : {fps:.2f}")
    print(f"Frames : {frame_count}")
    print(f"Size   : {width} x {height}")
    print()

    extractor = CameraLandmarkExtractor()

    all_features = []
    tracked_frames = 0
    left_frames = 0
    right_frames = 0

    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        features, result = extractor.extract(frame)

        all_features.append(features)

        left_found = bool(result.left_hand_landmarks)
        right_found = bool(result.right_hand_landmarks)

        if left_found:
            left_frames += 1

        if right_found:
            right_frames += 1

        if left_found or right_found:
            tracked_frames += 1

        frame_index += 1

        print(
            f"\rProcessing frame "
            f"{frame_index}/{frame_count}",
            end="",
            flush=True,
        )

    cap.release()

    print()
    print()

    motion = np.asarray(all_features, dtype=np.float32)

    output_path = (
        OUTPUT_DIR /
        f"{video_path.stem}_225.npy"
    )

    np.save(output_path, motion)

    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Motion shape : {motion.shape}")
    print(f"Saved to     : {output_path}")
    print()
    print(f"Frames with at least one hand : "
          f"{tracked_frames}/{frame_count}")
    print(f"Left hand detected            : "
          f"{left_frames}/{frame_count}")
    print(f"Right hand detected           : "
          f"{right_frames}/{frame_count}")
    print()

    if left_frames == 0 and right_frames == 0:
        print("WARNING: No hands were detected.")
    elif tracked_frames < frame_count * 0.7:
        print("WARNING: Hand tracking is missing in many frames.")
    else:
        print("Hand tracking coverage looks usable.")

    print("=" * 70)


if __name__ == "__main__":
    video = (
        Path(sys.argv[1]).expanduser()
        if len(sys.argv) > 1
        else DEFAULT_VIDEO
    )

    extract_video(video)
