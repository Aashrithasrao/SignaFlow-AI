import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="About — SignaFlow AI",
    page_icon="ℹ️",
    layout="wide",
)

# ============================================================
# CSS
# ============================================================

css_path = Path("frontend/styles.css")

if css_path.exists():
    st.html(
        f"<style>{css_path.read_text()}</style>",
    )

# ============================================================
# HEADER
# ============================================================

st.html(
    """
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
            ABOUT SIGNaflow
        </div>

    </div>
    """,
)

# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3 = st.columns(3)

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
    <div class="sf-page-hero">

        <div class="sf-page-eyebrow">
            ABOUT SIGNaflow
        </div>

        <div class="sf-page-title">
            Communication<br>
            should have no barriers.
        </div>

        <div class="sf-page-description">
            SignaFlow is a bidirectional artificial-intelligence
            platform designed to connect spoken communication
            and sign language through a single interactive system.
        </div>

    </div>
    """,
)

# ============================================================
# PROBLEM / APPROACH
# ============================================================

left, right = st.columns(2, gap="large")

with left:
    st.html(
        """
        <div class="sf-info-card">

            <div class="sf-info-number">
                THE PROBLEM
            </div>

            <div class="sf-info-title">
                Communication gaps
            </div>

            <div class="sf-info-text">
                Communication between people who use spoken
                language and people who communicate through
                sign language can be difficult when a shared
                communication method is unavailable.
            </div>

        </div>
        """,
    )

with right:
    st.html(
        """
        <div class="sf-info-card">

            <div class="sf-info-number">
                THE APPROACH
            </div>

            <div class="sf-info-title">
                One system. Two directions.
            </div>

            <div class="sf-info-text">
                SignaFlow approaches the problem from both
                directions: spoken language can become visual
                signs, while recognized signs can become speech.
            </div>

        </div>
        """,
    )

# ============================================================
# PRINCIPLES
# ============================================================

st.html(
    '<div class="sf-section-label">WHAT DEFINES SIGNaflow</div>',
)

principles = [
    (
        "01",
        "BIDIRECTIONAL",
        "Designed around both Speech → Sign and Sign → Speech.",
    ),
    (
        "02",
        "AI-POWERED",
        "Uses machine learning and computer vision for sign recognition.",
    ),
    (
        "03",
        "REAL-TIME",
        "Designed for interactive communication rather than offline translation.",
    ),
    (
        "04",
        "ACCESSIBILITY",
        "Focused on reducing communication barriers through technology.",
    ),
]

columns = st.columns(2)

for index, principle in enumerate(principles):

    number, title, description = principle

    with columns[index % 2]:

        st.html(
            f"""
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
            """,
        )

# ============================================================
# VISION
# ============================================================

st.html(
    """
    <div class="sf-page-hero" style="margin-top:25px">

        <div class="sf-page-eyebrow">
            OUR VISION
        </div>

        <div class="sf-page-title">
            Built with AI.<br>
            Designed for people.
        </div>

        <div class="sf-page-description">
            The goal of SignaFlow is not simply to recognize
            gestures or convert speech. It is to create a
            natural communication bridge where both directions
            can work together inside one accessible experience.
        </div>

    </div>
    """,
)

# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="sf-footer">

        <span>🤟 SignaFlow AI</span>

        <span>
            Communication without barriers.
        </span>

    </div>
    """,
)
