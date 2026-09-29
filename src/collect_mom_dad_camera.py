from pathlib import Path
import cv2
import numpy as np
import time

from src.camera_landmarks import CameraLandmarkExtractor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "camera_mom_dad"

MOM_DIR = OUTPUT_DIR / "mom"
DAD_DIR = OUTPUT_DIR / "dad"

TARGET_PER_CLASS = 150


def save_sequence(sequence, label, index):

    folder = MOM_DIR if label == "mom" else DAD_DIR
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"{label}_camera_{index:04d}.npy"

    array = np.asarray(
        sequence,
        dtype=np.float32
    )

    np.save(path, array)

    return path


def main():

    MOM_DIR.mkdir(parents=True, exist_ok=True)
    DAD_DIR.mkdir(parents=True, exist_ok=True)

    mom_count = len(list(MOM_DIR.glob("*.npy")))
    dad_count = len(list(DAD_DIR.glob("*.npy")))

    print("=" * 70)
    print("SIGNAFLOW AI — MOM / DAD CAMERA DATA COLLECTION")
    print("=" * 70)

    print()
    print(f"MOM already collected: {mom_count}/{TARGET_PER_CLASS}")
    print(f"DAD already collected: {dad_count}/{TARGET_PER_CLASS}")
    print()

    print("Controls:")
    print("  M = collect MOM")
    print("  D = collect DAD")
    print("  Q = quit")
    print()

    extractor = CameraLandmarkExtractor()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    sequence = []
    collecting = False
    current_label = None

    print("Camera started.")
    print()

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        landmarks, result = extractor.extract(frame)

        # ----------------------------------------------------
        # COLLECTING
        # ----------------------------------------------------

        if collecting:

            sequence.append(
                np.asarray(
                    landmarks,
                    dtype=np.float32
                )
            )

            required_frames = 32

            if len(sequence) >= required_frames:

                if current_label == "mom":
                    index = mom_count + 1

                    if index <= TARGET_PER_CLASS:
                        path = save_sequence(
                            sequence,
                            "mom",
                            index
                        )

                        mom_count += 1

                        print(
                            f"MOM {mom_count}/{TARGET_PER_CLASS} saved: "
                            f"{path.name}"
                        )

                elif current_label == "dad":
                    index = dad_count + 1

                    if index <= TARGET_PER_CLASS:
                        path = save_sequence(
                            sequence,
                            "dad",
                            index
                        )

                        dad_count += 1

                        print(
                            f"DAD {dad_count}/{TARGET_PER_CLASS} saved: "
                            f"{path.name}"
                        )

                sequence.clear()
                collecting = False
                current_label = None

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        if collecting:

            state_text = (
                f"COLLECTING {current_label.upper()} "
                f"{len(sequence)}/32"
            )

        else:

            state_text = "READY"

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 100),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            state_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"MOM: {mom_count}/{TARGET_PER_CLASS}   "
            f"DAD: {dad_count}/{TARGET_PER_CLASS}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "M = MOM   D = DAD   Q = QUIT",
            (20, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "SignaFlow - MOM/DAD Data Collection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if not collecting and key == ord("m"):

            if mom_count < TARGET_PER_CLASS:

                sequence.clear()
                current_label = "mom"
                collecting = True

                print(
                    "\nCollecting MOM..."
                )

            else:

                print(
                    "\nMOM already complete."
                )

        if not collecting and key == ord("d"):

            if dad_count < TARGET_PER_CLASS:

                sequence.clear()
                current_label = "dad"
                collecting = True

                print(
                    "\nCollecting DAD..."
                )

            else:

                print(
                    "\nDAD already complete."
                )

    cap.release()

    extractor.landmarker.close()

    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("COLLECTION FINISHED")
    print("=" * 70)
    print(f"MOM: {mom_count}/{TARGET_PER_CLASS}")
    print(f"DAD: {dad_count}/{TARGET_PER_CLASS}")
    print()


if __name__ == "__main__":
    main()
