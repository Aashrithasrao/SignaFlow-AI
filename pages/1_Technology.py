import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Technology — SignaFlow AI",
    page_icon="⚙️",
    layout="wide",
)

# ============================================================
# CSS
# ============================================================

css_path = Path("frontend/styles.css")

if css_path.exists():
    st.html(
        f"<style>{css_path.read_text()}</style>"
    )

# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="sf-nav">

    <div class="sf-brand">

        <div class="sf-logo">🤟</div>

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
        TECHNOLOGY
    </div>

</div>
""")

# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3 = st.columns(3)

with nav1:
    st.page_link(
        "app.py",
        label="Workspace",
        icon="🏠"
    )

with nav2:
    st.page_link(
        "pages/1_Technology.py",
        label="Technology",
        icon="⚙️"
    )

with nav3:
    st.page_link(
        "pages/2_About.py",
        label="About",
        icon="ℹ️"
    )

# ============================================================
# HERO
# ============================================================

st.html("""
<div class="sf-page-hero">

    <div class="sf-page-eyebrow">
        THE TECHNOLOGY
    </div>

    <div class="sf-page-title">
        Intelligence<br>
        behind the signs.
    </div>

    <div class="sf-page-description">
        SignaFlow combines speech recognition,
        language processing, landmark-based computer
        vision and sequence learning to create a
        bidirectional sign-language communication system.
    </div>

</div>
""")

# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

st.html("""
<div class="sf-section-label">
    SYSTEM ARCHITECTURE
</div>
""")

left, right = st.columns(2, gap="large")

with left:
    st.html("""
    <div class="sf-info-card">

        <div class="sf-info-number">
            PIPELINE 01
        </div>

        <div class="sf-info-title">
            🎤 Speech → Sign
        </div>

        <div class="sf-info-text">
            Spoken language is captured, converted into
            text, processed into an ASL-oriented sentence
            structure and mapped to available sign assets.
        </div>

    </div>
    """)

with right:
    st.html("""
    <div class="sf-info-card">

        <div class="sf-info-number">
            PIPELINE 02
        </div>

        <div class="sf-info-title">
            🤟 Sign → Speech
        </div>

        <div class="sf-info-text">
            Camera input is transformed into hand and body
            landmarks, resized into temporal sequences,
            classified by the trained model and converted
            into understandable speech.
        </div>

    </div>
    """)

# ============================================================
# SPEECH → SIGN PIPELINE
# ============================================================

st.html("""
<div class="sf-section-label">
    SPEECH → SIGN
</div>

<div class="sf-pipeline">

    <div class="sf-pipeline-step">
        🎤 Speech Recognition
    </div>

    <div class="sf-arrow">↓</div>

    <div class="sf-pipeline-step">
        📝 Text Normalization
    </div>

    <div class="sf-arrow">↓</div>

    <div class="sf-pipeline-step">
        🧠 ASL Grammar Processing
    </div>

    <div class="sf-arrow">↓</div>

    <div class="sf-pipeline-step">
        🔎 Sign Vocabulary
    </div>

    <div class="sf-arrow">↓</div>

    <div class="sf-pipeline-step">
        🎬 Sign Video Sequence
    </div>

</div>
""")

# ============================================================
# SIGN → SPEECH PIPELINE
# ============================================================

st.html("""
<div class="sf-section-label">
    SIGN → SPEECH
</div>

<div class="sf-pipeline">

    <div class="sf-pipeline-step">
        📷 Camera
    </div>

    <div class="sf-arrow">↓</div>

    <div class="sf-pipeline-step">
        ✋ Landmark Extraction
    </div>

    <div class="sf-arrow">↓</div>

    <div class="sf-pipeline-step">
        📐 32 × 225 Sequence
    </div>

    <div class="sf-arrow">↓</div>

    <div class="sf-pipeline-step">
        🧠 Bidirectional Sequence Model
    </div>

    <div class="sf-arrow">↓</div>

    <div class="sf-pipeline-step">
        🔤 50-Class Sign Recognition
    </div>

    <div class="sf-arrow">↓</div>

    <div class="sf-pipeline-step">
        🔊 Text-to-Speech
    </div>

</div>
""")

# ============================================================
# CORE TECHNOLOGIES
# ============================================================

st.html("""
<div class="sf-section-label">
    CORE TECHNOLOGIES
</div>
""")

technology = [
    (
        "01",
        "Python",
        "Core application logic and AI pipeline."
    ),
    (
        "02",
        "TensorFlow",
        "Sequence-model training and inference."
    ),
    (
        "03",
        "MediaPipe",
        "Visual landmark extraction."
    ),
    (
        "04",
        "Streamlit",
        "Interactive application interface."
    ),
]

columns = st.columns(4, gap="medium")

for column, (number, title, description) in zip(
    columns,
    technology
):
    with column:
        st.html(f"""
        <div class="sf-info-card">

            <div class="sf-info-number">
                {number}
            </div>

            <div class="sf-info-title">
                {title}
            </div>

            <div class="sf-info-text">
                {description}
            </div>

        </div>
        """)

# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="sf-footer">

    <span>
        🤟 SignaFlow AI
    </span>

    <span>
        Technology behind bidirectional communication
    </span>

</div>
""")
