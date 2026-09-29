import re
import sys
from pathlib import Path


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# =========================================================
# CURRENT SIGN VOCABULARY
# =========================================================

from config.sign_vocabulary import (
    SIGN_VOCABULARY
)


# =========================================================
# PHRASE NORMALIZATION
# =========================================================

PHRASE_MAP = {
    "thank you": "thankyou",
    "good morning": "morning",
    "good night": "night",
}


# =========================================================
# WORDS WE CAN IGNORE IN SIMPLE ASL GLOSS
# =========================================================

ARTICLES = {
    "a",
    "an",
    "the",
}


# =========================================================
# CLEAN ENGLISH
# =========================================================

def clean_text(text):

    text = text.lower().strip()

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Normalize common phrases
    for phrase, replacement in PHRASE_MAP.items():

        text = text.replace(
            phrase,
            replacement
        )

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    return text


# =========================================================
# CREATE SIMPLE ASL-ORIENTED GLOSS
# =========================================================

def create_gloss(text):

    text = clean_text(
        text
    )

    words = text.split()

    # -----------------------------------------------------
    # Remove articles
    # -----------------------------------------------------

    words = [
        word
        for word in words
        if word not in ARTICLES
    ]

    # -----------------------------------------------------
    # Greeting + self introduction
    #
    # "hi i am harshita"
    # → HI MY NAME HARSHITA
    # -----------------------------------------------------

    if (
        len(words) >= 4
        and words[0] in {
            "hi",
            "hello"
        }
        and words[1] == "i"
        and words[2] == "am"
    ):

        name = words[3]

        return [
            "hi",
            "my",
            "name",
            name
        ]


    # -----------------------------------------------------
    # "my name is harshita"
    # → MY NAME HARSHITA
    # -----------------------------------------------------

    if (
        len(words) >= 4
        and words[0] == "my"
        and words[1] == "name"
        and words[2] == "is"
    ):

        name = words[3]

        return [
            "my",
            "name",
            name
        ]


    # -----------------------------------------------------
    # "what is your name?"
    # → YOUR NAME WHAT
    # -----------------------------------------------------

    if (
        "what" in words
        and "your" in words
        and "name" in words
    ):

        return [
            "your",
            "name",
            "what"
        ]


    # -----------------------------------------------------
    # "how are you?"
    # → YOU HOW
    # -----------------------------------------------------

    if (
        "how" in words
        and "you" in words
    ):

        return [
            "you",
            "how"
        ]


    # -----------------------------------------------------
    # "where are you?"
    # → YOU WHERE
    # -----------------------------------------------------

    if (
        "where" in words
        and "you" in words
    ):

        return [
            "you",
            "where"
        ]


    # -----------------------------------------------------
    # SIMPLE GENERAL GLOSS
    # -----------------------------------------------------

    return words


# =========================================================
# FINGERSPELLING
# =========================================================

def create_fingerspelling(
    word
):

    letters = []

    for character in word:

        if character.isalpha():

            letters.append(
                character.upper()
            )

    return letters


# =========================================================
# DETECT POSSIBLE NAME
# =========================================================

def is_name(
    word,
    gloss,
    index
):

    # -----------------------------------------------------
    # Pattern:
    #
    # MY NAME HARSHITA
    #          ↑
    # -----------------------------------------------------

    if index >= 2:

        if (
            gloss[index - 2] == "my"
            and
            gloss[index - 1] == "name"
        ):

            return True

    return False


# =========================================================
# CLASSIFY GLOSS
# =========================================================

def classify_gloss(
    gloss
):

    sequence = []

    unsupported = []

    for index, word in enumerate(
        gloss
    ):

        # -------------------------------------------------
        # Name → fingerspell
        # -------------------------------------------------

        if is_name(
            word,
            gloss,
            index
        ):

            letters = create_fingerspelling(
                word
            )

            sequence.append({
                "word": word,
                "type": "fingerspell",
                "letters": letters
            })

            continue


        # -------------------------------------------------
        # Existing sign
        # -------------------------------------------------

        if word in SIGN_VOCABULARY:

            sequence.append({
                "word": word,
                "type": "sign",
                "asset": (
                    f"signs/{word}/"
                    f"{word}.mp4"
                )
            })

            continue


        # -------------------------------------------------
        # Unknown
        # -------------------------------------------------

        unsupported.append(
            word
        )

    return sequence, unsupported


# =========================================================
# COMPLETE PROCESSOR
# =========================================================

def process_sentence(
    text
):

    gloss = create_gloss(
        text
    )

    sequence, unsupported = (
        classify_gloss(
            gloss
        )
    )

    return {
        "original_text": text,
        "gloss": gloss,
        "sequence": sequence,
        "unsupported": unsupported
    }


# =========================================================
# DISPLAY
# =========================================================

def display_result(
    result
):

    print("\n" + "=" * 65)
    print("SPEECH → ASL SIGN SEQUENCE")
    print("=" * 65)

    print("\nOriginal English:")
    print(
        result["original_text"]
    )

    print("\nASL-oriented gloss:")

    print(
        " ".join(
            word.upper()
            for word in result["gloss"]
        )
    )

    print("\nSign sequence:")

    if not result["sequence"]:

        print(
            "  No supported signs."
        )

    else:

        for item in result["sequence"]:

            if item["type"] == "sign":

                print(
                    f"  🤟 {item['word']:<12}"
                    f" SIGN"
                )

            elif (
                item["type"]
                == "fingerspell"
            ):

                letters = " → ".join(
                    item["letters"]
                )

                print(
                    f"  🔤 {item['word']:<12}"
                    f" FINGERSPELL"
                    f" → {letters}"
                )

    print("\nUnsupported:")

    if result["unsupported"]:

        print(
            "  " +
            ", ".join(
                result["unsupported"]
            )
        )

    else:

        print(
            "  None"
        )

    print("=" * 65)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    text = input(
        "\nEnter an English sentence: "
    )

    result = process_sentence(
        text
    )

    display_result(
        result
    )
