import sys
import time
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# ============================================================
# PATHS & IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera_landmarks import CameraLandmarkExtractor
from src.preprocess_landmarks import normalize_sequence

# Model configurations
MODELS_CONFIG = {
    "1": {
        "name": "5-Class Core (94%+ High Precision)",
        "model_path": PROJECT_ROOT / "models" / "sign_model_5class_hybrid_cnn.keras",
        "classes_path": PROJECT_ROOT / "models" / "classes_5class.npy",
        "threshold": 0.45,
        "hints": [
            "HELLO: Wave hand near ear/head",
            "MOM:   Open 5 hand, thumb on chin",
            "DAD:   Open 5 hand, thumb on forehead",
            "WATER: 'W' three fingers tapping chin",
            "FOOD:  Pinched fingertips to mouth",
        ],
    },
    "2": {
        "name": "10-Class (hello, mom, dad, water, food, yes, no, please, go, home)",
        "model_path": PROJECT_ROOT / "models" / "sign_model_10class_hybrid_cnn.keras",
        "classes_path": PROJECT_ROOT / "models" / "classes_10class.npy",
        "threshold": 0.40,
        "hints": [
            "YES: Fist nodding up/down",
            "NO: Index+Middle tap Thumb twice (Duck beak)",
            "GO: Roll index fingers outward",
            "HOME: Flat-O taps mouth THEN ear (2 touches)",
            "PLEASE: Rub flat chest in circle",
        ],
    },
    "3": {
        "name": "15-Class Extended (Conversational Tier)",
        "model_path": PROJECT_ROOT / "models" / "sign_model_15class_hybrid_cnn.keras",
        "classes_path": PROJECT_ROOT / "models" / "classes_15class.npy",
        "threshold": 0.40,
        "hints": [
            "THANKYOU: Flat hand from chin forward",
            "DRINK: Cup hand tilted to mouth",
            "HAPPY: Open hand patting chest upward",
            "SAD: Open hand drawn down face",
            "HUNGRY: C-hand sliding down chest",
        ],
    },
    "4": {
        "name": "50-Class Full Vocabulary (78.5% Global)",
        "model_path": PROJECT_ROOT / "models" / "sign_model_50_hybrid_cnn.keras",
        "classes_path": PROJECT_ROOT / "dataset" / "processed" / "classes.npy",
        "threshold": 0.45,
        "hints": [
            "50 WLASL Signs Active",
            "Requires deliberate, clean signing motion",
        ],
    },
}

SEQUENCE_LENGTH = 32
FEATURES = 225

MIN_STABLE_FRAMES = 3
COOLDOWN_FRAMES = 15
PREDICTION_INTERVAL = 2


def load_model_mode(mode_key):
    cfg = MODELS_CONFIG[mode_key]
    model_path = cfg["model_path"]
    classes_path = cfg["classes_path"]

    # Fallback to 50-class if subset doesn't exist
    if not model_path.exists():
        print(f"Warning: {model_path.name} not found yet. Using 50-class fallback.")
        model_path = PROJECT_ROOT / "models" / "sign_model_50_hybrid_cnn.keras"
        classes_path = PROJECT_ROOT / "dataset" / "processed" / "classes.npy"

    print(f"\n[MODEL SWITCH]: Loading {cfg['name']} ({model_path.name})...")
    model = tf.keras.models.load_model(str(model_path))
    classes = np.load(classes_path, allow_pickle=True)
    classes = [str(c) for c in classes]
    return model, classes, cfg


def draw_hud(frame, current_sign, confidence, top3, sentence, fps, buffer_len, latency_ms, mode_cfg, hands_present):
    h, w = frame.shape[:2]

    # Semi-transparent overlay panels
    overlay = frame.copy()
    # Top header bar
    cv2.rectangle(overlay, (0, 0), (w, 115), (16, 18, 26), -1)
    # Bottom sentence bar
    cv2.rectangle(overlay, (0, h - 90), (w, h), (16, 18, 26), -1)
    # Right-side candidates panel
    cv2.rectangle(overlay, (w - 260, 125), (w - 10, 275), (22, 26, 36), -1)
    # Left-side hint panel
    cv2.rectangle(overlay, (10, 125), (320, 260), (22, 26, 36), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # 1. Header & Active Mode
    cv2.putText(
        frame,
        f"SIGNAFLOW — {mode_cfg['name'].upper()}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 230, 255),
        2,
        cv2.LINE_AA,
    )

    # 2. Performance Stats & Hands Status
    hands_str = "HANDS: DETECTED" if hands_present else "HANDS: NONE (IDLE)"
    hands_color = (0, 255, 120) if hands_present else (120, 120, 120)
    perf_text = f"FPS: {fps:.1f}  |  LATENCY: {latency_ms:.1f}ms  |  BUF: {buffer_len}/{SEQUENCE_LENGTH}  |  "
    cv2.putText(
        frame,
        perf_text,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 190, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        hands_str,
        (380, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        hands_color,
        2,
        cv2.LINE_AA,
    )

    # 3. Active Detected Sign
    if not hands_present:
        sign_text = "STATE: IDLE (Raise hand to sign)"
        sign_color = (130, 130, 130)
    elif current_sign and confidence >= mode_cfg["threshold"]:
        sign_color = (0, 255, 120) if confidence >= 0.70 else (0, 220, 255)
        sign_text = f"ACTIVE SIGN: {current_sign.upper()} ({confidence * 100:.1f}%)"
    else:
        sign_color = (180, 180, 180)
        sign_text = "STATE: LISTENING..."

    cv2.putText(
        frame,
        sign_text,
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        sign_color,
        2,
        cv2.LINE_AA,
    )

    # 4. Gesture Hints (Left Panel)
    cv2.putText(
        frame,
        "GESTURE GUIDE",
        (20, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 200, 100),
        1,
        cv2.LINE_AA,
    )
    for i, hint in enumerate(mode_cfg["hints"][:5]):
        cv2.putText(
            frame,
            hint[:35],
            (20, 172 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (220, 225, 230),
            1,
            cv2.LINE_AA,
        )

    # 5. Top-3 Predictions (Right Panel)
    cv2.putText(
        frame,
        "TOP PREDICTIONS",
        (w - 250, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 200, 100),
        1,
        cv2.LINE_AA,
    )
    for i, (name, prob) in enumerate(top3):
        color = (0, 255, 120) if i == 0 and prob >= mode_cfg["threshold"] and hands_present else (200, 200, 200)
        cv2.putText(
            frame,
            f"{i+1}. {name[:12]:<12} {prob*100:4.1f}%",
            (w - 250, 178 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA,
        )

    # 6. Composed Sentence (Bottom Panel)
    cv2.putText(
        frame,
        "COMPOSED SENTENCE:",
        (20, h - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (150, 200, 255),
        1,
        cv2.LINE_AA,
    )
    sentence_str = " ".join(sentence) if sentence else "(Perform sign gestures to build sentence...)"
    sentence_color = (255, 255, 255) if sentence else (130, 130, 130)
    cv2.putText(
        frame,
        sentence_str[:65],
        (20, h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        sentence_color,
        2,
        cv2.LINE_AA,
    )

    # 7. Navigation Controls
    cv2.putText(
        frame,
        "[1] 5-Class  [2] 10-Class  [3] 15-Class  [4] 50-Class  |  [C] Clear  [S] Speak  [Q] Quit",
        (w - 600, h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (160, 170, 185),
        1,
        cv2.LINE_AA,
    )


def speak_sentence(text):
    if not text:
        return
    try:
        import subprocess
        subprocess.Popen(["say", text])
    except Exception:
        pass


def main():
    print("=" * 70)
    print("SIGNAFLOW AI — HYBRID 1D-CNN LIVE CAMERA TEST BENCH")
    print("=" * 70)

    current_mode_key = "1"  # Default to 5-Class for rock-solid testing
    model, classes, mode_cfg = load_model_mode(current_mode_key)

    print("\nInitializing MediaPipe Holistic Landmarker...")
    extractor = CameraLandmarkExtractor()

    print("\nOpening Camera (Index 0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Please grant camera permission.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    recent_predictions = deque(maxlen=4)
    sentence = []
    last_committed_sign = None
    cooldown = 0

    frame_count = 0
    fps = 0.0
    latency_ms = 0.0
    fps_timer = time.time()
    fps_frame_counter = 0

    current_sign = ""
    current_conf = 0.0
    top3 = []

    print("\n" + "=" * 70)
    print("LIVE TESTING ACTIVE!")
    print("  • Press '1' for 5-Class Model (HELLO, MOM, DAD, WATER, FOOD)")
    print("  • Press '2' for 10-Class Model (Adds YES, NO, PLEASE, GO, HOME)")
    print("  • Press '3' for 15-Class Model (Adds THANKYOU, DRINK, HAPPY, SAD, HUNGRY)")
    print("  • Press '4' for 50-Class Full Vocabulary")
    print("  • Press 'c' to Clear Sentence")
    print("  • Press 's' to Speak Aloud")
    print("  • Press 'q' or 'ESC' to Exit")
    print("=" * 70 + "\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read camera frame.")
            break

        frame_count += 1
        fps_frame_counter += 1
        frame = cv2.flip(frame, 1)

        # 1. Extract 225 landmarks
        features, result = extractor.extract(frame)
        sequence_buffer.append(features)

        # Check if hands are actively present in view
        hands_present = bool(result.left_hand_landmarks or result.right_hand_landmarks)

        # 2. Run inference when buffer is full and hands are visible
        if len(sequence_buffer) == SEQUENCE_LENGTH and (frame_count % PREDICTION_INTERVAL == 0):
            if hands_present:
                seq_array = np.asarray(sequence_buffer, dtype=np.float32)
                seq_norm = normalize_sequence(seq_array)
                input_tensor = np.expand_dims(seq_norm, axis=0)

                t0 = time.time()
                probs = model(input_tensor, training=False).numpy()[0]
                latency_ms = (time.time() - t0) * 1000

                # ====================================================
                # ANATOMICAL HANDSHAPE DISAMBIGUATION: YES vs NO vs GO
                # ====================================================
                # 1. YES (Fist): Index, Middle, Ring, Pinky all curled tight to palm.
                # 2. GO  (Pointing): Index extended forward, Middle & Pinky curled to palm.
                # 3. NO  (Duck Beak): Index & Middle extended together, tapping Thumb.
                active_hands = [h for h in [result.right_hand_landmarks, result.left_hand_landmarks] if h]
                if active_hands:
                    yes_idx = classes.index("yes") if "yes" in classes else -1
                    no_idx = classes.index("no") if "no" in classes else -1
                    go_idx = classes.index("go") if "go" in classes else -1
                    please_idx = classes.index("please") if "please" in classes else -1

                    detected_shape = None  # "fist", "pointing", "duck_beak", "flat_chest"
                    for hand_lms in active_hands:
                        try:
                            # Scale-invariant palm length (wrist 0 to middle MCP 9)
                            palm_scale = ((hand_lms[9].x - hand_lms[0].x)**2 + (hand_lms[9].y - hand_lms[0].y)**2)**0.5
                            if palm_scale < 1e-4:
                                continue

                            # Finger extensions: distance from MCP (base knuckle) to fingertip / palm_scale
                            index_ext = ((hand_lms[8].x - hand_lms[5].x)**2 + (hand_lms[8].y - hand_lms[5].y)**2)**0.5 / palm_scale
                            mid_ext = ((hand_lms[12].x - hand_lms[9].x)**2 + (hand_lms[12].y - hand_lms[9].y)**2)**0.5 / palm_scale
                            pinky_ext = ((hand_lms[20].x - hand_lms[17].x)**2 + (hand_lms[20].y - hand_lms[17].y)**2)**0.5 / palm_scale

                            # Distances to thumb tip (4)
                            d_index_thumb = ((hand_lms[8].x - hand_lms[4].x)**2 + (hand_lms[8].y - hand_lms[4].y)**2)**0.5 / palm_scale
                            d_mid_thumb = ((hand_lms[12].x - hand_lms[4].x)**2 + (hand_lms[12].y - hand_lms[4].y)**2)**0.5 / palm_scale

                            # 1. Closed Fist (YES): All fingers curled
                            if index_ext < 0.68 and mid_ext < 0.68 and pinky_ext < 0.72:
                                detected_shape = "fist"
                                break

                            # 2. Pointing Forward / Outward (GO): Index extended, Middle curled
                            elif index_ext > 0.75 and mid_ext < 0.65 and pinky_ext < 0.70:
                                detected_shape = "pointing"
                                break

                            # 3. Duck Beak (NO): Index AND Middle extended together, tapping Thumb tip
                            elif index_ext > 0.65 and mid_ext > 0.65 and (d_index_thumb < 0.65 or d_mid_thumb < 0.65):
                                detected_shape = "duck_beak"
                                break

                            # 4. Flat Hand on Chest (PLEASE): Open palm on chest
                            elif index_ext > 0.70 and mid_ext > 0.70 and pinky_ext > 0.70 and d_index_thumb > 0.45:
                                if result.pose_landmarks:
                                    chin_y = result.pose_landmarks[0].y + 0.08
                                    if hand_lms[0].y > chin_y:
                                        detected_shape = "flat_chest"
                                        break
                                else:
                                    detected_shape = "flat_chest"
                                    break
                        except Exception:
                            pass

                    # Apply precise mutually exclusive adjustments:
                    if detected_shape == "fist" and yes_idx >= 0:
                        probs[yes_idx] = max(probs[yes_idx] * 2.2, 0.45)
                        if no_idx >= 0: probs[no_idx] *= 0.15
                        if go_idx >= 0: probs[go_idx] *= 0.15
                        probs = probs / np.sum(probs)

                    elif detected_shape == "pointing" and go_idx >= 0:
                        probs[go_idx] = max(probs[go_idx] * 2.2, 0.45)
                        if no_idx >= 0: probs[no_idx] *= 0.15
                        if yes_idx >= 0: probs[yes_idx] *= 0.15
                        probs = probs / np.sum(probs)

                    elif detected_shape == "duck_beak" and no_idx >= 0:
                        probs[no_idx] = max(probs[no_idx] * 2.2, 0.45)
                        if yes_idx >= 0: probs[yes_idx] *= 0.15
                        if go_idx >= 0: probs[go_idx] *= 0.15
                        probs = probs / np.sum(probs)

                    elif detected_shape == "flat_chest" and please_idx >= 0:
                        top3_indices = np.argsort(probs)[::-1][:3]
                        if probs[please_idx] >= 0.12 or (please_idx in top3_indices):
                            probs[please_idx] = max(probs[please_idx] * 2.0, 0.45)
                            probs = probs / np.sum(probs)

                top_indices = np.argsort(probs)[::-1][:min(3, len(classes))]
                top3 = [(classes[idx], float(probs[idx])) for idx in top_indices]

                top_idx = top_indices[0]
                current_conf = float(probs[top_idx])
                current_sign = classes[top_idx]

                # 3. Stability filter for sentence formation
                eff_threshold = 0.28 if current_sign in ["no", "go", "yes", "please"] else mode_cfg["threshold"]
                if current_conf >= eff_threshold:
                    recent_predictions.append(current_sign)
                else:
                    recent_predictions.append(None)

                if cooldown > 0:
                    cooldown -= 1
                else:
                    if len(recent_predictions) >= 2:
                        counts = Counter([p for p in recent_predictions if p is not None])
                        if counts:
                            best_candidate, best_count = counts.most_common(1)[0]
                            # FOOD requires 4 consecutive frames (0.26s hold at mouth) so
                            # the passing motion of HOME (mouth to ear) does not get falsely hijacked by FOOD!
                            # NO, GO, YES, PLEASE commit cleanly in 2 frames
                            req_frames = 4 if best_candidate == "food" else (2 if best_candidate in ["no", "go", "yes", "please"] else 3)
                            if best_count >= req_frames:
                                if best_candidate != last_committed_sign:
                                    sentence.append(best_candidate.upper())
                                    last_committed_sign = best_candidate
                                    cooldown = COOLDOWN_FRAMES
                                    print(f"[SENTENCE COMMITTED]: {best_candidate.upper()} ({current_conf * 100:.1f}%) -> Current: {' '.join(sentence)}")
            else:
                # Hands not present: stay idle, don't generate false signs
                current_sign = ""
                current_conf = 0.0
                recent_predictions.clear()
                latency_ms = 0.0

        # Calculate FPS
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps = fps_frame_counter / elapsed
            fps_frame_counter = 0
            fps_timer = time.time()

        # 4. Draw HUD
        draw_hud(
            frame,
            current_sign,
            current_conf,
            top3,
            sentence,
            fps,
            len(sequence_buffer),
            latency_ms,
            mode_cfg,
            hands_present,
        )

        cv2.imshow("SignaFlow AI — Hybrid 1D-CNN Local Camera Test", frame)

        # 5. Handle Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), 27]:
            print("\nExiting camera test...")
            break
        elif key == ord("c"):
            sentence.clear()
            last_committed_sign = None
            print("[SENTENCE CLEARED]")
        elif key == ord("s"):
            sentence_text = " ".join(sentence)
            print(f"[SPEAKING]: {sentence_text}")
            speak_sentence(sentence_text)
        elif key in [ord("1"), ord("2"), ord("3"), ord("4")]:
            new_mode = chr(key)
            if new_mode != current_mode_key:
                current_mode_key = new_mode
                model, classes, mode_cfg = load_model_mode(current_mode_key)
                sequence_buffer.clear()
                recent_predictions.clear()
                top3 = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
