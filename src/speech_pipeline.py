import re

from src.sentence_processor import translate_to_asl_gloss
from src.text_to_sign import create_sign_sequence


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_input(text):
    """
    Clean raw speech-recognition text.
    """

    if not text:
        return ""

    text = text.strip().lower()

    text = re.sub(
        r"[^\w\s']",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# SPEECH → SIGN PIPELINE
# ============================================================

def run_speech_to_sign(text):
    """
    Complete Speech → Sign pipeline.

    Flow:

        Speech text
             ↓
        Text cleaning
             ↓
        ASL sentence processing
             ↓
        ASL gloss
             ↓
        Sign sequence generation
             ↓
        Sign videos / fingerspelling
    """

    # --------------------------------------------------------
    # STEP 1 — CLEAN INPUT
    # --------------------------------------------------------

    original = clean_input(text)

    if not original:

        return {
            "original": "",
            "normalized": [],
            "gloss_text": "",
            "sequence": [],
            "unsupported": [],
        }

    # --------------------------------------------------------
    # STEP 2 — ASL SENTENCE PROCESSING
    # --------------------------------------------------------

    gloss = translate_to_asl_gloss(
        original
    )

    # translate_to_asl_gloss()
    # returns something like:
    #
    # ['HI', 'MY', 'NAME', 'AASHRITHA']

    if isinstance(gloss, list):

        normalized_words = []

        for word in gloss:

            word = str(word).strip().lower()

            if word:
                normalized_words.append(word)

    else:

        normalized_words = [
            word.strip().lower()
            for word in str(gloss).split()
            if word.strip()
        ]

    # --------------------------------------------------------
    # STEP 3 — CREATE GLOSS TEXT
    # --------------------------------------------------------

    gloss_text = " ".join(
        normalized_words
    )

    # --------------------------------------------------------
    # STEP 4 — TEXT → SIGN
    # --------------------------------------------------------

    sign_result = create_sign_sequence(
        gloss_text
    )

    # --------------------------------------------------------
    # STEP 5 — EXTRACT RESULT
    # --------------------------------------------------------

    if not isinstance(
        sign_result,
        dict
    ):

        sign_result = {
            "sequence": [],
            "unsupported": normalized_words,
        }

    sequence = sign_result.get(
        "sequence",
        []
    )

    unsupported = sign_result.get(
        "unsupported",
        []
    )

    # --------------------------------------------------------
    # STEP 6 — RETURN COMPLETE RESULT
    # --------------------------------------------------------

    return {
        "original": original,

        "normalized": normalized_words,

        "gloss_text": gloss_text,

        "sequence": sequence,

        "unsupported": unsupported,
    }


# ============================================================
# DISPLAY FINGER SPELLING
# ============================================================

def format_fingerspelling(letters):
    """
    Convert fingerspelling data into readable text.

    Supports both:

        ["H", "A", "I"]

    and:

        [
            {"letter": "H"},
            {"letter": "A"},
            {"letter": "I"}
        ]
    """

    display_letters = []

    for item in letters:

        # ----------------------------------------------------
        # Dictionary format
        # ----------------------------------------------------

        if isinstance(
            item,
            dict
        ):

            value = (
                item.get("letter")
                or item.get("word")
                or item.get("asset")
                or str(item)
            )

        # ----------------------------------------------------
        # Normal string format
        # ----------------------------------------------------

        else:

            value = str(item)

        display_letters.append(
            value
        )

    return " → ".join(
        display_letters
    )


# ============================================================
# PRINT PIPELINE RESULT
# ============================================================

def print_result(result):

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    print("\nOriginal:")

    print(
        result.get(
            "original",
            ""
        )
    )

    # --------------------------------------------------------
    # ASL SEQUENCE
    # --------------------------------------------------------

    print("\nASL-style sequence:")

    normalized = result.get(
        "normalized",
        []
    )

    if normalized:

        print(
            " → ".join(
                word.upper()
                for word in normalized
            )
        )

    else:

        print("(empty)")

    # --------------------------------------------------------
    # SIGN SEQUENCE
    # --------------------------------------------------------

    print("\nSign sequence:")

    sequence = result.get(
        "sequence",
        []
    )

    if not sequence:

        print(
            "  No supported signs."
        )

    else:

        for item in sequence:

            word = item.get(
                "word",
                ""
            )

            sign_type = item.get(
                "type",
                ""
            )

            # ----------------------------------------------
            # NORMAL SIGN
            # ----------------------------------------------

            if sign_type == "sign":

                asset = item.get(
                    "asset",
                    ""
                )

                print(
                    f"  {word} → SIGN → {asset}"
                )

            # ----------------------------------------------
            # FINGERSPELLING
            # ----------------------------------------------

            elif sign_type == "fingerspell":

                letters = item.get(
                    "letters",
                    []
                )

                formatted = format_fingerspelling(
                    letters
                )

                print(
                    f"  {word} → FINGERSPELL → "
                    f"{formatted}"
                )

            # ----------------------------------------------
            # UNKNOWN TYPE
            # ----------------------------------------------

            else:

                print(
                    f"  {word} → {sign_type}"
                )

    # --------------------------------------------------------
    # UNSUPPORTED WORDS
    # --------------------------------------------------------

    print("\nUnsupported:")

    unsupported = result.get(
        "unsupported",
        []
    )

    if unsupported:

        print(
            "  "
            + ", ".join(
                unsupported
            )
        )

    else:

        print(
            "  None"
        )

    print("\n" + "=" * 60)


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SPEECH → SIGN PIPELINE TEST")
    print("=" * 60)

    text = input(
        "\nEnter sentence: "
    ).strip()

    result = run_speech_to_sign(
        text
    )

    print_result(
        result
    )

    print(
        "\nPIPELINE TEST COMPLETE"
    )
