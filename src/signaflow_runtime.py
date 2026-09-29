from pathlib import Path

import base64
import hashlib
import json
import subprocess

import speech_recognition as sr

from src.sentence_processor import (
    translate_to_asl_gloss,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

SIGNS_DIR = (
    PROJECT_ROOT / "signs"
)

RUNTIME_DIR = (
    PROJECT_ROOT / "runtime"
)

RUNTIME_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# WORD ALIASES
# ============================================================

WORD_ALIASES = {
    "hi": "hello",
    "mom": "mother",
    "dad": "father",
    "thank": "thankyou",
    "thanks": "thankyou",
}


# ============================================================
# FIND SIGN VIDEO
# ============================================================

def find_sign_video(word: str):

    word = (
        str(word)
        .strip()
        .lower()
    )

    if not word:
        return None

    # --------------------------------------------------------
    # Resolve aliases
    # --------------------------------------------------------

    folder_word = WORD_ALIASES.get(
        word,
        word
    )

    # --------------------------------------------------------
    # Candidate locations
    # --------------------------------------------------------

    folder = (
        SIGNS_DIR
        / folder_word
    )

    candidates = [

        # Preferred cleaned video
        folder
        / f"{folder_word}_clean.mp4",

        # Original video
        folder
        / f"{folder_word}.mp4",

        # Clean-folder fallback
        SIGNS_DIR
        / "clean"
        / f"{folder_word}_clean.mp4",
    ]

    for candidate in candidates:

        if (
            candidate.exists()
            and candidate.is_file()
            and candidate.stat().st_size > 0
        ):

            return candidate

    return None


# ============================================================
# FINGERSPELLING
# ============================================================

def get_fingerspelling_videos(word):

    word = (
        str(word)
        .strip()
    )

    if not word.isalpha():
        return []

    finger_dir = (
        SIGNS_DIR
        / "fingerspelling"
    )

    letter_paths = []

    for letter in word.upper():

        path = (
            finger_dir
            / f"{letter}.mp4"
        )

        if (
            not path.exists()
            or path.stat().st_size <= 0
        ):

            return []

        letter_paths.append(
            path
        )

    return letter_paths


# ============================================================
# SPEECH AUDIO → TEXT
# ============================================================

def recognize_audio(uploaded_audio):

    if uploaded_audio is None:
        return ""

    raw = uploaded_audio.getvalue()

    if not raw:
        return ""

    input_path = (
        RUNTIME_DIR
        / "speech_input.bin"
    )

    wav_path = (
        RUNTIME_DIR
        / "speech_input.wav"
    )

    input_path.write_bytes(
        raw
    )

    # --------------------------------------------------------
    # Convert browser audio → 16 kHz mono WAV
    # --------------------------------------------------------

    command = [

        "ffmpeg",

        "-y",

        "-i",

        str(input_path),

        "-ar",

        "16000",

        "-ac",

        "1",

        str(wav_path),
    ]

    result = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Could not convert recorded audio.\n\n"
            + result.stderr[-1500:]
        )

    recognizer = sr.Recognizer()

    with sr.AudioFile(
        str(wav_path)
    ) as source:

        audio_data = (
            recognizer.record(
                source
            )
        )

    try:

        return recognizer.recognize_google(
            audio_data
        )

    except sr.UnknownValueError:

        return ""

    except sr.RequestError as error:

        raise RuntimeError(
            f"Speech recognition service error: {error}"
        )


# ============================================================
# SPEECH → SIGN
# ============================================================

def process_speech_text(text: str):

    text = (
        text or ""
    ).strip()

    if not text:

        return {
            "original": "",
            "gloss": [],
            "gloss_text": "",
            "videos": [],
            "unsupported": [],
        }

    # --------------------------------------------------------
    # English → ASL-style gloss
    # --------------------------------------------------------

    gloss = (
        translate_to_asl_gloss(
            text
        )
    )

    normalized = []

    for word in gloss:

        word = (
            str(word)
            .strip()
            .lower()
        )

        if word:

            normalized.append(
                word
            )

    videos = []

    unsupported = []

    # ========================================================
    # RESOLVE EVERY GLOSS TOKEN
    # ========================================================

    for word in normalized:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # A single alphabetic character ALWAYS means
        # fingerspelling.
        #
        # This prevents the letter "I" from being interpreted
        # as the normal "I" sign when spelling a person's name.
        #
        # Example:
        #
        # AASHRITHA
        #
        # A A S H R I T H A
        #
        # The I here must use:
        #
        # signs/fingerspelling/I.mp4
        #
        # NOT:
        #
        # signs/i/i_clean.mp4
        # ----------------------------------------------------

        if (
            len(word) == 1
            and word.isalpha()
        ):

            letter_paths = (
                get_fingerspelling_videos(
                    word
                )
            )

            if letter_paths:

                videos.append({

                    "word": word,

                    "letter": word.upper(),

                    "path": str(
                        letter_paths[0]
                    ),

                    "type": "fingerspell",

                })

                continue

        # ----------------------------------------------------
        # 1. Exact normal sign video
        # ----------------------------------------------------

        video = find_sign_video(
            word
        )

        if video is not None:

            videos.append({

                "word": word,

                "path": str(
                    video
                ),

                "type": "sign",

            })

            continue

        # ----------------------------------------------------
        # 2. Fingerspelling fallback
        # ----------------------------------------------------

        letter_paths = (
            get_fingerspelling_videos(
                word
            )
        )

        if letter_paths:

            for letter, path in zip(
                word.upper(),
                letter_paths
            ):

                videos.append({

                    "word": word,

                    "letter": letter,

                    "path": str(
                        path
                    ),

                    "type": "fingerspell",

                })

            continue

        # ----------------------------------------------------
        # 3. Truly unsupported
        # ----------------------------------------------------

        unsupported.append(
            word
        )

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "original": text,

        "gloss": normalized,

        "gloss_text": " ".join(
            normalized
        ),

        "videos": videos,

        "unsupported": unsupported,
    }


# ============================================================
# VIDEO → DATA URI
# ============================================================

def video_to_data_uri(
    path: str | Path
):

    path = Path(
        path
    )

    if not path.exists():
        return None

    data = path.read_bytes()

    encoded = (
        base64
        .b64encode(data)
        .decode("ascii")
    )

    return (
        "data:video/mp4;base64,"
        + encoded
    )


# ============================================================
# MACOS TEXT → SPEECH
# ============================================================

def text_to_speech(
    text: str
):

    text = (
        text or ""
    ).strip()

    if not text:
        return None

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]

    aiff_path = (
        RUNTIME_DIR
        / f"speech_{digest}.aiff"
    )

    wav_path = (
        RUNTIME_DIR
        / f"speech_{digest}.wav"
    )

    if not wav_path.exists():

        subprocess.run(

            [
                "say",
                "-o",
                str(aiff_path),
                text,
            ],

            check=True,
        )

        subprocess.run(

            [
                "ffmpeg",
                "-y",
                "-i",
                str(aiff_path),
                "-ar",
                "44100",
                "-ac",
                "1",
                str(wav_path),
            ],

            check=True,

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,
        )

    return wav_path


# ============================================================
# LIVE SIGN RESULT
# ============================================================

SIGN_RESULT_FILE = (
    RUNTIME_DIR
    / "sign_result.json"
)


def read_sign_result():

    if not SIGN_RESULT_FILE.exists():

        return {

            "words": [],

            "sentence": "",

            "last_sign": "",

            "confidence": 0.0,

            "status": "WAITING",
        }

    try:

        return json.loads(
            SIGN_RESULT_FILE.read_text()
        )

    except Exception:

        return {

            "words": [],

            "sentence": "",

            "last_sign": "",

            "confidence": 0.0,

            "status": "WAITING",
        }


def reset_sign_result():

    SIGN_RESULT_FILE.write_text(

        json.dumps(

            {

                "words": [],

                "sentence": "",

                "last_sign": "",

                "confidence": 0.0,

                "status": "WAITING",

            },

            indent=2,

        )

    )
