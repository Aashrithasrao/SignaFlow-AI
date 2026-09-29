import re
import sys
from pathlib import Path

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.sign_vocabulary import SIGN_VOCABULARY


# ---------------------------------------------------------
# Words that indicate a person's name
# ---------------------------------------------------------

NAME_INDICATORS = {
    "name",
}


# ---------------------------------------------------------
# Clean text
# ---------------------------------------------------------

def clean_text(text):

    text = text.lower()

    phrase_map = {
        "thank you": "thankyou",
        "good morning": "morning",
        "good night": "night",
    }

    for phrase, replacement in phrase_map.items():
        text = text.replace(
            phrase,
            replacement
        )

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    return text.split()


# ---------------------------------------------------------
# Detect whether a word should be fingerspelled
# ---------------------------------------------------------

def should_fingerspell(
    word,
    previous_words
):

    # If the previous word was "name",
    # treat the next word as a possible name.
    if previous_words:

        if previous_words[-1] in NAME_INDICATORS:
            return True

    return False


# ---------------------------------------------------------
# Create fingerspelling sequence
# ---------------------------------------------------------

def create_fingerspell_sequence(word):

    letters = []

    for letter in word.lower():

        if letter.isalpha():

            letters.append({
                "letter": letter.upper(),
                "type": "fingerspell",
                "asset": (
                    f"signs/fingerspelling/"
                    f"{letter.upper()}.mp4"
                )
            })

    return letters


# ---------------------------------------------------------
# Create complete sign sequence
# ---------------------------------------------------------

def create_sign_sequence(text):

    words = clean_text(text)

    sequence = []
    unsupported = []

    previous_words = []

    for word in words:

        # ---------------------------------------------
        # Normal sign
        # ---------------------------------------------

        if word in SIGN_VOCABULARY:

            sequence.append({
                "word": word,
                "type": "sign",
                "asset": (
                    f"signs/{word}/"
                    f"{word}.mp4"
                )
            })

        # ---------------------------------------------
        # Possible name
        # ---------------------------------------------

        elif should_fingerspell(
            word,
            previous_words
        ):

            sequence.append({
                "word": word,
                "type": "fingerspell",
                "letters": (
                    create_fingerspell_sequence(
                        word
                    )
                )
            })

        # ---------------------------------------------
        # Unknown word
        # ---------------------------------------------

        else:

            unsupported.append(word)

        previous_words.append(word)

    return {
        "original_text": text,
        "sequence": sequence,
        "unsupported": unsupported,
    }


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------

def print_result(result):

    print("\n" + "=" * 60)
    print("TEXT → SIGN ANALYSIS")
    print("=" * 60)

    print("\nOriginal:")
    print(result["original_text"])

    print("\nSign sequence:")

    if not result["sequence"]:

        print("  No sign sequence created.")

    for item in result["sequence"]:

        if item["type"] == "sign":

            print(
                f"  {item['word']} "
                f"→ SIGN "
                f"→ {item['asset']}"
            )

        elif item["type"] == "fingerspell":

            letters = " → ".join(
                x["letter"]
                for x in item["letters"]
            )

            print(
                f"  {item['word']} "
                f"→ FINGERSPELL "
                f"→ {letters}"
            )

    print("\nUnsupported words:")

    if result["unsupported"]:

        print(
            "  " +
            ", ".join(
                result["unsupported"]
            )
        )

    else:

        print("  None")

    print("=" * 60)


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    text = input(
        "Enter a sentence: "
    )

    result = create_sign_sequence(
        text
    )

    print_result(result)
