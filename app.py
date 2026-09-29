import base64
import html
import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# SIGNaflow AI — WORKSPACE
# ============================================================

st.set_page_config(
    page_title="SignaFlow AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CSS_PATH = PROJECT_ROOT / "frontend" / "styles.css"

RUNTIME_DIR = PROJECT_ROOT / "runtime"

RESULT_PATH = (
    RUNTIME_DIR / "sign_result.json"
)


# ============================================================
# SESSION STATE
# ============================================================

if "mode" not in st.session_state:
    st.session_state.mode = "speech"

if "sign_process" not in st.session_state:
    st.session_state.sign_process = None

if "sign_running" not in st.session_state:
    st.session_state.sign_running = False

if "speech_result" not in st.session_state:
    st.session_state.speech_result = None


# ============================================================
# LOAD CSS
# ============================================================

if CSS_PATH.exists():

    st.html(
        f"<style>{CSS_PATH.read_text()}</style>"
    )


# ============================================================
# SIGN → SPEECH BACKEND
# ============================================================

def start_sign_recognition():

    try:

        RUNTIME_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Clear old result.

        with open(
            RESULT_PATH,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "sentence": "",
                    "status": "STARTING",
                    "current_sign": "",
                    "confidence": 0.0,
                    "timestamp": time.time(),
                },
                f,
            )

        old_process = (
            st.session_state.get(
                "sign_process"
            )
        )

        if old_process is not None:

            if old_process.poll() is None:

                st.session_state.sign_running = True

                return

        win_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        unix_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        if win_python.exists():
            executable = str(win_python)
        elif unix_python.exists():
            executable = str(unix_python)
        else:
            executable = sys.executable

        process = subprocess.Popen(
            [
                executable,
                "-m",
                "src.sign_to_speech_live",
            ],
            cwd=str(PROJECT_ROOT),
        )

        st.session_state.sign_process = process

        st.session_state.sign_running = True

    except Exception as e:

        st.session_state.sign_running = False

        st.error(
            f"Could not start recognition: {e}"
        )


def stop_sign_recognition():

    process = (
        st.session_state.get(
            "sign_process"
        )
    )

    if process is not None:

        try:

            if process.poll() is None:

                process.terminate()

                try:

                    process.wait(
                        timeout=2
                    )

                except subprocess.TimeoutExpired:

                    process.kill()

        except Exception:
            pass

    st.session_state.sign_running = False


def check_sign_process():

    process = (
        st.session_state.get(
            "sign_process"
        )
    )

    if process is None:
        return

    if process.poll() is not None:

        st.session_state.sign_running = False


def read_sign_result():

    default = {
        "sentence": "",
        "status": "STOPPED",
        "current_sign": "",
        "confidence": 0.0,
        "timestamp": 0,
    }

    if not RESULT_PATH.exists():

        return default

    try:

        with open(
            RESULT_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        if not isinstance(
            data,
            dict,
        ):

            return default

        return {
            "sentence": data.get(
                "sentence",
                "",
            ),
            "status": data.get(
                "status",
                "STOPPED",
            ),
            "current_sign": data.get(
                "current_sign",
                "",
            ),
            "confidence": float(
                data.get(
                    "confidence",
                    0.0,
                )
            ),
            "timestamp": data.get(
                "timestamp",
                0,
            ),
        }

    except Exception:

        return default


# ============================================================
# SPEECH → SIGN BACKEND
# ============================================================

def process_speech_audio(audio):

    from src.signaflow_runtime import (
        recognize_audio,
        process_speech_text,
    )

    text = recognize_audio(audio)

    if not text:
        return None

    result = process_speech_text(
        text
    )

    if not isinstance(
        result,
        dict,
    ):

        result = {
            "recognized_text": text,
            "videos": [],
            "unsupported": [],
            "gloss_text": "",
        }

    result["recognized_text"] = text

    return result


# ============================================================
# VIDEO PATH RESOLVER
# ============================================================

def resolve_video_path(item):
    """
    Accept the different asset formats used by the backend.

    Examples accepted:

        Path(...)
        "signs/water/water_clean.mp4"

        {
            "path": "...",
            "word": "water"
        }

        {
            "video": "...",
            "word": "water"
        }

        {
            "asset_path": "...",
            "word": "water"
        }
    """

    # --------------------------------------------------------
    # Plain Path
    # --------------------------------------------------------

    if isinstance(
        item,
        Path,
    ):

        path = item

        if not path.is_absolute():

            path = PROJECT_ROOT / path

        if path.exists():

            return path

        return None


    # --------------------------------------------------------
    # Plain string
    # --------------------------------------------------------

    if isinstance(
        item,
        str,
    ):

        path = Path(item)

        if not path.is_absolute():

            path = PROJECT_ROOT / path

        if path.exists():

            return path

        return None


    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        item,
        dict,
    ):

        possible_keys = [
            "path",
            "video",
            "video_path",
            "asset_path",
            "file",
            "file_path",
            "filepath",
        ]

        for key in possible_keys:

            value = item.get(key)

            if value is None:
                continue

            # Recursive handling.

            resolved = resolve_video_path(
                value
            )

            if resolved is not None:

                return resolved

        return None


    return None


# ============================================================
# VIDEO WORD RESOLVER
# ============================================================

def resolve_video_word(
    item,
    path,
):

    if isinstance(
        item,
        dict,
    ):

        for key in [
            "word",
            "sign",
            "label",
            "name",
            "text",
        ]:

            value = item.get(key)

            if value:

                return str(value)

    # Try parent folder.

    try:

        return path.parent.name

    except Exception:

        return path.stem


# ============================================================
# VIDEO DATA URI
# ============================================================

def path_to_video_data_uri(
    video_path,
):

    path = resolve_video_path(
        video_path
    )

    if path is None:
        return None

    try:

        encoded = base64.b64encode(
            path.read_bytes()
        ).decode(
            "utf-8"
        )

        return (
            "data:video/mp4;base64,"
            + encoded
        )

    except Exception:

        return None


# ============================================================
# SEQUENTIAL VIDEO PLAYER
# ============================================================

def render_sequential_videos(
    videos,
):

    if not videos:

        st.info(
            "No sign videos available."
        )

        return


    # --------------------------------------------------------
    # Normalize backend video objects
    # --------------------------------------------------------

    playable = []

    for item in videos:

        path = resolve_video_path(
            item
        )

        if path is None:

            continue

        data_uri = (
            path_to_video_data_uri(
                path
            )
        )

        if data_uri is None:

            continue

        word = resolve_video_word(
            item,
            path,
        )

        playable.append(
            {
                "src": data_uri,
                "word": str(word),
            }
        )


    # --------------------------------------------------------
    # Nothing playable
    # --------------------------------------------------------

    if not playable:

        st.warning(
            "The sign sequence was created, "
            "but no playable MP4 assets were found."
        )

        return


    # --------------------------------------------------------
    # Prepare JSON
    # --------------------------------------------------------

    videos_json = json.dumps(
        playable
    )


    # --------------------------------------------------------
    # Player
    # --------------------------------------------------------

    player_html = f"""
    <!DOCTYPE html>

    <html>

    <head>

    <style>

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 0;
            background: transparent;
            color: white;

            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "SF Pro Display",
                "SF Pro Text",
                Arial,
                sans-serif;
        }}

        .player {{
            width: 100%;
            overflow: hidden;

            border-radius: 22px;

            background:
                linear-gradient(
                    145deg,
                    rgba(255,255,255,0.08),
                    rgba(255,255,255,0.025)
                );

            border:
                1px solid
                rgba(255,255,255,0.10);

            box-shadow:
                0 20px 60px
                rgba(0,0,0,0.28);
        }}

        video {{
            width: 100%;
            height: 430px;

            display: block;

            background: #08090b;

            object-fit: contain;
        }}

        .info {{
            display: flex;

            justify-content:
                space-between;

            align-items: center;

            padding:
                15px 18px 10px;
        }}

        .word {{
            font-size: 18px;

            font-weight: 650;

            letter-spacing:
                0.05em;
        }}

        .counter {{
            color: #8e949e;

            font-size: 13px;
        }}

        .dots {{
            display: flex;

            gap: 6px;

            padding:
                4px 18px 17px;

            flex-wrap: wrap;
        }}

        .dot {{
            height: 4px;

            flex: 1;

            min-width: 20px;

            border-radius: 99px;

            background:
                rgba(255,255,255,0.13);
        }}

        .dot.active {{
            background:
                rgba(255,255,255,0.85);
        }}

    </style>

    </head>

    <body>

    <div class="player">

        <video
            id="signVideo"
            autoplay
            playsinline
            controls
        ></video>

        <div class="info">

            <div
                id="word"
                class="word"
            >
                —
            </div>

            <div
                id="counter"
                class="counter"
            >
                1 / 1
            </div>

        </div>

        <div
            id="dots"
            class="dots"
        ></div>

    </div>


    <script>

        const sequence =
            {videos_json};

        let currentIndex = 0;

        const video =
            document.getElementById(
                "signVideo"
            );

        const word =
            document.getElementById(
                "word"
            );

        const counter =
            document.getElementById(
                "counter"
            );

        const dots =
            document.getElementById(
                "dots"
            );


        function createDots() {{

            dots.innerHTML = "";

            sequence.forEach(
                (_, index) => {{

                    const dot =
                        document.createElement(
                            "div"
                        );

                    dot.className =
                        "dot";

                    dot.id =
                        "dot-" + index;

                    dots.appendChild(
                        dot
                    );
                }}
            );
        }}


        function updateDots() {{

            sequence.forEach(
                (_, index) => {{

                    const dot =
                        document.getElementById(
                            "dot-" + index
                        );

                    if (!dot) {{
                        return;
                    }}

                    if (
                        index ===
                        currentIndex
                    ) {{

                        dot.classList.add(
                            "active"
                        );

                    }} else {{

                        dot.classList.remove(
                            "active"
                        );
                    }}
                }}
            );
        }}


        function playCurrent() {{

            if (
                !sequence.length
            ) {{

                return;
            }}

            const current =
                sequence[
                    currentIndex
                ];

            video.src =
                current.src;

            word.textContent =
                String(
                    current.word || "SIGN"
                ).toUpperCase();

            counter.textContent =
                (
                    currentIndex + 1
                )
                + " / "
                + sequence.length;

            updateDots();

            video.currentTime = 0;

            video.play().catch(
                () => {{}}
            );
        }}


        video.addEventListener(
            "ended",
            () => {{

                currentIndex++;

                if (
                    currentIndex >=
                    sequence.length
                ) {{

                    currentIndex = 0;
                }}

                playCurrent();

            }}
        );


        createDots();

        playCurrent();

    </script>

    </body>

    </html>
    """


    components.html(
        player_html,
        height=520,
        scrolling=False,
    )


# ============================================================
# TOP HEADER
# ============================================================

st.html(
    """
    <div class="sf-nav">

        <div class="sf-brand">

            <div class="sf-logo">
                🤟
            </div>

            <div>

                <div class="sf-brand-name">
                    SignaFlow
                </div>

                <div class="sf-brand-sub">
                    AI COMMUNICATION
                </div>

            </div>

        </div>

        <div class="sf-online">

            <span class="sf-dot"></span>

            SYSTEM READY

        </div>

    </div>
    """
)


# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3 = st.columns(
    [1, 1, 1]
)

with nav1:

    st.page_link(
        "app.py",
        label="Workspace",
        icon="🏠",
    )

with nav2:

    st.page_link(
        "pages/1_Technology.py",
        label="Technology",
        icon="⚙️",
    )

with nav3:

    st.page_link(
        "pages/2_About.py",
        label="About",
        icon="ℹ️",
    )


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="sf-hero">

        <div>

            <div class="sf-eyebrow">
                BIDIRECTIONAL SIGN LANGUAGE AI
            </div>

            <div class="sf-hero-title">

                Communication
                <br>

                <span class="sf-gradient">
                    without barriers.
                </span>

            </div>

            <div class="sf-hero-description">

                SignaFlow connects spoken language and
                sign language through artificial intelligence,
                creating a bidirectional communication experience.

            </div>

            <div class="sf-tags">

                <span class="sf-tag">
                    REAL-TIME
                </span>

                <span class="sf-tag">
                    LANDMARK AI
                </span>

                <span class="sf-tag">
                    ASL PROCESSING
                </span>

            </div>

        </div>

        <div class="sf-hero-visual">

            <div class="sf-glow"></div>

            <div class="sf-hand">
                🤟
            </div>

        </div>

    </div>
    """
)


# ============================================================
# EXPERIENCE SELECTOR
# ============================================================

st.html(
    """
    <div class="sf-section-label">
        CHOOSE YOUR EXPERIENCE
    </div>
    """
)


speech_col, sign_col = st.columns(
    2,
    gap="large",
)


# ============================================================
# SPEECH → SIGN MODE
# ============================================================

with speech_col:

    active = (
        st.session_state.mode
        == "speech"
    )

    active_text = (
        '<div class="sf-active-label">'
        '● ACTIVE MODE'
        '</div>'
        if active
        else ""
    )

    st.html(
        f"""
        <div class="sf-mode-card
            {"sf-mode-active" if active else ""}">

            <div class="sf-mode-icon">
                🎤
            </div>

            <div class="sf-mode-title">
                SPEECH → SIGN
            </div>

            <div class="sf-mode-description">
                Talk naturally and convert your sentence
                into visual sign language.
            </div>

            {active_text}

        </div>
        """
    )

    if st.button(
        "Use Speech → Sign",
        key="speech_mode",
        type=(
            "primary"
            if active
            else "secondary"
        ),
        width="stretch",
    ):

        st.session_state.mode = (
            "speech"
        )

        st.rerun()


# ============================================================
# SIGN → SPEECH MODE
# ============================================================

with sign_col:

    active = (
        st.session_state.mode
        == "sign"
    )

    active_text = (
        '<div class="sf-active-label">'
        '● ACTIVE MODE'
        '</div>'
        if active
        else ""
    )

    st.html(
        f"""
        <div class="sf-mode-card
            {"sf-mode-active" if active else ""}">

            <div class="sf-mode-icon">
                🤟
            </div>

            <div class="sf-mode-title">
                SIGN → SPEECH
            </div>

            <div class="sf-mode-description">
                Use your gestures and let AI recognize
                the sign and convert it into speech.
            </div>

            {active_text}

        </div>
        """
    )

    if st.button(
        "Use Sign → Speech",
        key="sign_mode",
        type=(
            "primary"
            if active
            else "secondary"
        ),
        width="stretch",
    ):

        st.session_state.mode = (
            "sign"
        )

        st.rerun()


# ============================================================
# SPEECH → SIGN WORKSPACE
# ============================================================

if (
    st.session_state.mode
    == "speech"
):

    st.html(
        """
        <div class="sf-section-label">
            SPEECH TO SIGN
        </div>

        <div class="sf-section-title">

            Speak naturally.

            <span style="color:#646a75">
                We'll handle the signs.
            </span>

        </div>
        """
    )


    voice_col, visual_col = st.columns(
        [0.9, 1.1],
        gap="large",
    )


    # ========================================================
    # VOICE STUDIO
    # ========================================================

    with voice_col:

        st.html(
            """
            <div class="sf-panel">

                <div class="sf-panel-header">

                    <div>

                        <div class="sf-panel-label">
                            VOICE STUDIO
                        </div>

                        <div class="sf-panel-title">
                            Speak your sentence
                        </div>

                    </div>

                    <div class="sf-panel-label">
                        READY
                    </div>

                </div>

                <div class="sf-center">

                    <div class="sf-circle">
                        🎙️
                    </div>

                    <div class="sf-center-title">
                        Record your message
                    </div>

                    <div class="sf-center-sub">
                        Speak naturally. Your audio will be
                        passed to the speech-processing pipeline.
                    </div>

                </div>

            </div>
            """
        )


        audio = st.audio_input(
            "Start recording",
            key="speech_audio",
        )


        if audio:

            st.audio(audio)

            with st.spinner(
                "Understanding your speech..."
            ):

                try:

                    result = (
                        process_speech_audio(
                            audio
                        )
                    )

                    st.session_state.speech_result = (
                        result
                    )

                except Exception as e:

                    st.session_state.speech_result = (
                        None
                    )

                    st.error(
                        "Speech processing failed: "
                        + str(e)
                    )


    # ========================================================
    # VISUAL SIGN PLAYER
    # ========================================================

    with visual_col:

        result = (
            st.session_state.speech_result
        )


        if result:

            recognized_text = (
                result.get(
                    "recognized_text",
                    "",
                )
            )

            gloss_text = (
                result.get(
                    "gloss_text",
                    "",
                )
            )

            videos = (
                result.get(
                    "videos",
                    [],
                )
            )

            unsupported = (
                result.get(
                    "unsupported",
                    [],
                )
            )


            st.html(
                """
                <div class="sf-panel">

                    <div class="sf-panel-header">

                        <div>

                            <div class="sf-panel-label">
                                SIGN VISUALIZER
                            </div>

                            <div class="sf-panel-title">
                                Visual sign sequence
                            </div>

                        </div>

                        <div class="sf-panel-label">
                            READY
                        </div>

                    </div>

                </div>
                """
            )


            st.markdown(
                "### You said"
            )

            st.write(
                recognized_text
            )


            st.markdown(
                "### ASL Gloss"
            )

            st.write(
                gloss_text
            )


            if videos:

                render_sequential_videos(
                    videos
                )

            else:

                st.info(
                    "No sign videos were found "
                    "for this sentence."
                )


            if unsupported:

                st.warning(
                    "Missing sign assets: "
                    + ", ".join(
                        str(x)
                        for x in unsupported
                    )
                )


        else:

            st.html(
                """
                <div class="sf-panel">

                    <div class="sf-panel-header">

                        <div>

                            <div class="sf-panel-label">
                                SIGN VISUALIZER
                            </div>

                            <div class="sf-panel-title">
                                Visual sign sequence
                            </div>

                        </div>

                        <div class="sf-panel-label">
                            READY
                        </div>

                    </div>

                    <div class="sf-video">

                        <div>

                            <div class="sf-video-icon">
                                ✨
                            </div>

                            <div class="sf-video-text">
                                Your sign video will appear here
                            </div>

                        </div>

                    </div>

                </div>
                """
            )


# ============================================================
# SIGN → SPEECH WORKSPACE
# ============================================================

else:

    check_sign_process()

    result = read_sign_result()

    status = result.get(
        "status",
        "STOPPED",
    )

    sentence = result.get(
        "sentence",
        "",
    )

    current_sign = result.get(
        "current_sign",
        "",
    )

    confidence = result.get(
        "confidence",
        0.0,
    )


    st.html(
        """
        <div class="sf-section-label">
            SIGN TO SPEECH
        </div>

        <div class="sf-section-title">

            Sign naturally.

            <span style="color:#646a75">
                AI will understand.
            </span>

        </div>

        <div style="background: rgba(30, 41, 59, 0.65); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 12px; padding: 12px 18px; margin: 16px 0 20px 0; display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 22px;">⚡</span>
                <div>
                    <div style="font-weight: 600; font-size: 14px; color: #f8fafc; letter-spacing: 0.3px;">
                        ACTIVE MODEL: DEEP HYBRID 1D-CNN + BiLSTM (50 CLASSES)
                    </div>
                    <div style="font-size: 12px; color: #94a3b8; margin-top: 2px;">
                        Loaded: <code style="color:#a5b4fc; background:rgba(99,102,241,0.15); padding:2px 6px; border-radius:4px;">sign_model_50_hybrid_cnn.keras</code> • 50 ASL Signs Ready
                    </div>
                </div>
            </div>
            <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.6); color: #34d399; font-size: 11px; font-weight: 700; padding: 5px 12px; border-radius: 20px; letter-spacing: 0.5px;">
                ● 50 CLASSES CONNECTED
            </div>
        </div>
        """
    )

    with st.expander("📋 View All 50 Recognized ASL Signs & Tips", expanded=False):
        st.markdown(
            """
            **50 Supported ASL Vocabulary Words:**
            `bad`, `better`, `book`, `boy`, `close`, `dad`, `drink`, `every`, `fine`, `first`,
            `food`, `girl`, `give`, `go`, `happy`, `have`, `hear`, `hello`, `home`, `hungry`,
            `later`, `listen`, `look`, `man`, `mom`, `morning`, `night`, `no`, `now`, `open`,
            `outside`, `please`, `read`, `sad`, `same`, `say`, `see`, `sleep`, `stay`, `talk`,
            `thankyou`, `thirsty`, `tomorrow`, `wait`, `wake`, `water`, `where`, `who`, `why`, `yes`

            *Tips for best recognition:*
            - **NO**: Index and middle fingers tap the thumb tip together (duck beak motion).
            - **YES**: Make a fist (S-hand) and nod up and down.
            - **GO**: Point index finger outward/forward away from body.
            - **PLEASE**: Flat open hand rubbing clockwise in a circle on chest.
            - **MOM / DAD**: Open 5-hand with thumb touching chin (MOM) or forehead (DAD).
            """
        )

    camera_col, recognition_col = st.columns(
        [1.1, 0.9],
        gap="large",
    )


    # ========================================================
    # CAMERA
    # ========================================================

    with camera_col:

        camera_status = (
            "LIVE"
            if st.session_state.sign_running
            else "READY"
        )


        st.html(
            f"""
            <div class="sf-panel">

                <div class="sf-panel-header">

                    <div>

                        <div class="sf-panel-label">
                            LIVE CAMERA
                        </div>

                        <div class="sf-panel-title">
                            Gesture input
                        </div>

                    </div>

                    <div class="sf-panel-label">
                        {camera_status}
                    </div>

                </div>

                <div class="sf-camera">

                    <div class="sf-camera-content">

                        <div class="sf-camera-icon">
                            🤟
                        </div>

                        <div>

                            <div>
                                {
                                    "Recognition active"
                                    if st.session_state.sign_running
                                    else "Camera ready"
                                }
                            </div>

                            <div class="sf-camera-sub">

                                {
                                    "The live camera window is open with the 50-Class AI model tracking landmarks."
                                    if st.session_state.sign_running
                                    else "Press Start Live Recognition below to open the camera window."
                                }

                            </div>

                        </div>

                    </div>

                </div>

            </div>
            """
        )


        start_col, stop_col = st.columns(
            2,
            gap="small",
        )


        with start_col:

            if st.button(
                "▶ Start Live Recognition",
                key="start_live",
                type="primary",
                width="stretch",
                disabled=(
                    st.session_state.sign_running
                ),
            ):

                start_sign_recognition()

                st.rerun()


        with stop_col:

            if st.button(
                "■ Stop Recognition",
                key="stop_live",
                width="stretch",
                disabled=(
                    not st.session_state.sign_running
                ),
            ):

                stop_sign_recognition()

                st.rerun()


    # ========================================================
    # RECOGNITION
    # ========================================================

    with recognition_col:

        if status == "RUNNING":

            status_text = "● LIVE"

        elif status == "AUTO_STOPPED":

            status_text = "AUTO STOPPED"

        elif status == "ERROR":

            status_text = "ERROR"

        else:

            status_text = "READY"


        safe_sign = html.escape(
            (
                str(
                    current_sign
                ).upper()
            )
            if current_sign
            else "—"
        )


        safe_sentence = html.escape(
            (
                str(sentence)
                if sentence
                else
                "Your recognized sentence will appear here."
            )
        )


        confidence_percent = (
            max(
                0.0,
                min(
                    1.0,
                    float(
                        confidence
                    ),
                ),
            )
            * 100
        )


        st.html(
            f"""
            <div class="sf-panel">

                <div class="sf-panel-header">

                    <div>

                        <div class="sf-panel-label">
                            AI RECOGNITION
                        </div>

                        <div class="sf-panel-title">
                            Sign interpretation
                        </div>

                    </div>

                    <div class="sf-panel-label">
                        {status_text}
                    </div>

                </div>


                <div class="sf-recognition">

                    <div class="sf-panel-label">
                        DETECTED SIGN
                    </div>

                    <div class="sf-recognized">
                        {safe_sign}
                    </div>

                    <div class="sf-confidence">
                        Confidence:
                        {confidence_percent:.1f}%
                    </div>

                </div>


                <div class="sf-transcript">

                    <div class="sf-panel-label">
                        SENTENCE
                    </div>

                    <div style="
                        margin-top:8px;
                        color:#b8bdc7;
                        font-size:18px;
                        line-height:1.5;
                    ">
                        {safe_sentence}
                    </div>

                </div>

            </div>
            """
        )


        # ====================================================
        # SPEAK SENTENCE
        # ====================================================

        if sentence:

            if st.button(
                "🔊 Speak Sentence",
                key="speak_result",
                type="primary",
                width="stretch",
            ):

                try:

                    from src.signaflow_runtime import (
                        text_to_speech,
                    )

                    with st.spinner(
                        "Preparing speech..."
                    ):

                        audio_path = (
                            text_to_speech(
                                sentence
                            )
                        )


                    if audio_path:

                        st.audio(
                            str(
                                audio_path
                            ),
                            format="audio/wav",
                        )

                    else:

                        st.warning(
                            "Could not generate speech."
                        )


                except Exception as e:

                    st.error(
                        "Text-to-speech failed: "
                        + str(e)
                    )


        else:

            st.button(
                "🔊 Speak Sentence",
                key="speak_result_disabled",
                disabled=True,
                width="stretch",
            )


# ============================================================
# AUTO REFRESH — SIGN MODE
# ============================================================

if (
    st.session_state.mode
    == "sign"
    and
    st.session_state.sign_running
):

    time.sleep(
        0.7
    )

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="sf-footer">

        <span>
            🤟 SignaFlow AI
        </span>

        <span>
            Bidirectional sign-language communication
        </span>

    </div>
    """
)
