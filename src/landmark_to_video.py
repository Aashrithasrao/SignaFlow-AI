import cv2
import numpy as np
import pandas as pd

from pathlib import Path
import subprocess
import tempfile


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "rendered_signs"

FPS = 15

WIDTH = 900
HEIGHT = 700

BACKGROUND = (18, 18, 24)

# MediaPipe-style drawing colors
POSE_POINT = (180, 180, 180)
HAND_POINT = (220, 220, 220)

LEFT_HAND_LINE = (120, 190, 255)
RIGHT_HAND_LINE = (255, 180, 120)
POSE_LINE = (150, 150, 150)


# ============================================================
# MEDIAPIPE CONNECTIONS
# ============================================================

# Hand landmark connections
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17),
]


# Simplified body pose connections
POSE_CONNECTIONS = [
    # face/head
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    # shoulders
    (11, 12),

    # left arm
    (11, 13),
    (13, 15),

    # right arm
    (12, 14),
    (14, 16),

    # torso
    (11, 23),
    (12, 24),
    (23, 24),

    # left leg
    (23, 25),
    (25, 27),

    # right leg
    (24, 26),
    (26, 28),

    # feet
    (27, 29),
    (29, 31),

    (28, 30),
    (30, 32),
]


# ============================================================
# HELPERS
# ============================================================

def clean_type(value):
    """
    Normalize landmark type names.
    """

    return str(value).strip().lower().replace(" ", "_")


def load_parquet(path):
    """
    Load raw WLASL landmark parquet.
    """

    df = pd.read_parquet(path)

    required = {
        "frame",
        "type",
        "landmark_index",
        "x",
        "y",
        "z",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["type"] = df["type"].map(clean_type)

    return df


def get_landmarks(df, frame, landmark_type, count):
    """
    Return count x 3 landmark array.

    Missing landmarks are zeros.
    """

    result = np.zeros(
        (count, 3),
        dtype=np.float32,
    )

    subset = df[
        (df["frame"] == frame)
        & (df["type"] == landmark_type)
    ]

    for _, row in subset.iterrows():

        try:
            index = int(row["landmark_index"])

            if 0 <= index < count:

                result[index] = [
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                ]

        except Exception:
            continue

    return result


def valid_point(point):
    """
    Check whether a landmark exists.
    """

    return not np.allclose(
        point,
        0.0,
        atol=1e-7,
    )


def smooth_landmarks(frames):
    """
    Light temporal smoothing.

    This makes the rendered animation less jittery.
    """

    if len(frames) < 2:
        return frames

    result = []

    previous = None

    alpha = 0.65

    for current in frames:

        current = current.copy()

        if previous is not None:

            valid = (
                np.linalg.norm(current, axis=1) > 1e-7
            )

            current[valid] = (
                alpha * current[valid]
                + (1.0 - alpha) * previous[valid]
            )

        result.append(current)

        previous = current

    return result


def resize_sequence(frames, target_length=32):
    """
    Uniformly resample frames to 32 frames.

    This makes the animation consistent with the
    model's 32-frame representation.
    """

    if not frames:
        return []

    if len(frames) == target_length:
        return frames

    indices = np.linspace(
        0,
        len(frames) - 1,
        target_length,
    )

    output = []

    for index in indices:

        lower = int(np.floor(index))
        upper = int(np.ceil(index))

        if lower == upper:

            output.append(
                frames[lower].copy()
            )

        else:

            ratio = index - lower

            frame = (
                frames[lower] * (1.0 - ratio)
                + frames[upper] * ratio
            )

            output.append(frame)

    return output


def normalize_for_display(frame):
    """
    Convert normalized landmark coordinates into
    attractive screen coordinates.

    x and y are expected to be approximately 0..1.
    """

    points = frame[:, :2].copy()

    valid = np.linalg.norm(
        points,
        axis=1,
    ) > 1e-7

    if not np.any(valid):
        return None

    valid_points = points[valid]

    min_x = np.min(valid_points[:, 0])
    max_x = np.max(valid_points[:, 0])

    min_y = np.min(valid_points[:, 1])
    max_y = np.max(valid_points[:, 1])

    # Prevent zero-size bounding boxes
    if max_x - min_x < 0.01:
        max_x = min_x + 0.01

    if max_y - min_y < 0.01:
        max_y = min_y + 0.01

    # Add padding
    padding = 0.15

    min_x -= padding
    max_x += padding

    min_y -= padding
    max_y += padding

    # Keep values reasonable
    min_x = min(min_x, 0.0)
    max_x = max(max_x, 1.0)

    min_y = min(min_y, 0.0)
    max_y = max(max_y, 1.0)

    screen = np.zeros_like(points)

    screen[:, 0] = (
        (points[:, 0] - min_x)
        / (max_x - min_x)
    )

    screen[:, 1] = (
        (points[:, 1] - min_y)
        / (max_y - min_y)
    )

    screen[:, 0] *= WIDTH
    screen[:, 1] *= HEIGHT

    return screen


def draw_skeleton(
    canvas,
    landmarks,
    start,
    count,
    connections,
    line_color,
    point_color,
):
    """
    Draw one landmark group.
    """

    group = landmarks[
        start:start + count
    ]

    points = normalize_for_display(
        landmarks
    )

    if points is None:
        return

    group_points = points[
        start:start + count
    ]

    # Draw connections
    for a, b in connections:

        if a >= len(group):
            continue

        if b >= len(group):
            continue

        if not valid_point(group[a]):
            continue

        if not valid_point(group[b]):
            continue

        p1 = tuple(
            np.int32(group_points[a])
        )

        p2 = tuple(
            np.int32(group_points[b])
        )

        cv2.line(
            canvas,
            p1,
            p2,
            line_color,
            3,
            cv2.LINE_AA,
        )

    # Draw points
    for i, point in enumerate(group):

        if not valid_point(point):
            continue

        x, y = np.int32(
            group_points[i]
        )

        cv2.circle(
            canvas,
            (x, y),
            6,
            point_color,
            -1,
            cv2.LINE_AA,
        )


def render_frames(frames, output_path):
    """
    Render landmark frames to MP4.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Temporary OpenCV video
    temp_path = output_path.with_suffix(
        ".temporary.mp4"
    )

    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not open video writer."
        )

    for frame_index, landmarks in enumerate(frames):

        canvas = np.zeros(
            (HEIGHT, WIDTH, 3),
            dtype=np.uint8,
        )

        canvas[:] = BACKGROUND

        # ----------------------------------------------------
        # The 225 landmarks are:
        #
        # 0-20     left hand
        # 21-41    right hand
        # 42-74    pose
        # ----------------------------------------------------

        # Left hand
        draw_skeleton(
            canvas,
            landmarks,
            start=0,
            count=21,
            connections=HAND_CONNECTIONS,
            line_color=LEFT_HAND_LINE,
            point_color=HAND_POINT,
        )

        # Right hand
        draw_skeleton(
            canvas,
            landmarks,
            start=21,
            count=21,
            connections=HAND_CONNECTIONS,
            line_color=RIGHT_HAND_LINE,
            point_color=HAND_POINT,
        )

        # Pose
        draw_skeleton(
            canvas,
            landmarks,
            start=42,
            count=33,
            connections=POSE_CONNECTIONS,
            line_color=POSE_LINE,
            point_color=POSE_POINT,
        )

        # Header
        cv2.putText(
            canvas,
            "SignaFlow AI",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "Landmark Animation",
            (30, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"Frame {frame_index + 1}/{len(frames)}",
            (WIDTH - 190, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        writer.write(canvas)

    writer.release()

    # --------------------------------------------------------
    # Convert to H.264 for browser/Streamlit compatibility
    # --------------------------------------------------------

    ffmpeg_available = (
        subprocess.run(
            ["which", "ffmpeg"],
            capture_output=True,
        ).returncode
        == 0
    )

    if ffmpeg_available:

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_path),
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

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:

            print(
                "WARNING: FFmpeg conversion failed."
            )

            temp_path.replace(output_path)

        else:

            temp_path.unlink(
                missing_ok=True
            )

    else:

        temp_path.replace(output_path)

    return output_path


# ============================================================
# MAIN RENDER FUNCTION
# ============================================================

def parquet_to_video(
    parquet_path,
    output_path=None,
):
    """
    Convert one raw WLASL Parquet file into
    a playable landmark animation.
    """

    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(
            parquet_path
        )

    print("=" * 70)
    print("SIGNAFLOW AI — PARQUET LANDMARK RENDERER")
    print("=" * 70)

    print(f"Input : {parquet_path}")

    df = load_parquet(
        parquet_path
    )

    print(
        f"Rows  : {len(df)}"
    )

    print(
        "Types :",
        sorted(df["type"].unique())
    )

    frames = []

    frame_numbers = sorted(
        df["frame"].unique()
    )

    for frame_number in frame_numbers:

        left_hand = get_landmarks(
            df,
            frame_number,
            "left_hand",
            21,
        )

        right_hand = get_landmarks(
            df,
            frame_number,
            "right_hand",
            21,
        )

        pose = get_landmarks(
            df,
            frame_number,
            "pose",
            33,
        )

        combined = np.concatenate(
            [
                left_hand,
                right_hand,
                pose,
            ],
            axis=0,
        )

        frames.append(
            combined
        )

    print(
        f"Original frames : {len(frames)}"
    )

    if not frames:
        raise RuntimeError(
            "No hand/pose frames found."
        )

    # Smooth
    frames = smooth_landmarks(
        frames
    )

    # Convert to 32 frames
    frames = resize_sequence(
        frames,
        target_length=32,
    )

    print(
        f"Rendered frames : {len(frames)}"
    )

    if output_path is None:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            OUTPUT_DIR
            / f"{parquet_path.stem}_landmarks.mp4"
        )

    output_path = Path(
        output_path
    )

    result = render_frames(
        frames,
        output_path,
    )

    print()
    print("SUCCESS")
    print(f"Output: {result}")
    print("=" * 70)

    return result


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Convert a SignaFlow/WLASL "
            "landmark Parquet file "
            "into an animated MP4."
        )
    )

    parser.add_argument(
        "parquet",
        help="Path to the parquet file.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output MP4 path.",
    )

    args = parser.parse_args()

    parquet_to_video(
        args.parquet,
        args.output,
    )
