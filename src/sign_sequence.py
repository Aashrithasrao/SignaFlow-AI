import re
import sys
from pathlib import Path


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from config.sign_vocabulary import SIGN_VOCABULARY


# =========================================================
# BASIC CONFIGURATION
# =========================================================

PHRASE_MAP = {
    "thank you": "thankyou",
    "good morning": "morning",
    "good night": "night",
}


ARTICLES = {
    "a",
    "an",
    "the",
}


FILLERS = {
    "uh",
    "um",
    "hmm",
    "actually",
    "basically",
}


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.lower().strip()

    for phrase, replacement in PHRASE_MAP.items():
        text = text.replace(
            phrase,
            replacement
        )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in ARTICLES
        and word not in FILLERS
    ]

    return words


# =========================================================
# HELPER
# =========================================================

def contains(words, word):

    return word in words


# =========================================================
# INTRODUCTION
# =========================================================

def detect_introduction(words):

    # ---------------------------------------------
    # "hi i am harshita"
    # ---------------------------------------------

    if (
        len(words) >= 4
        and words[0] in {"hi", "hello"}
        and words[1] == "i"
        and words[2] == "am"
    ):

        name = words[3]

        return [
            words[0],
            "my",
            "name",
            name
        ]

    # ---------------------------------------------
    # "my name is harshita"
    # ---------------------------------------------

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

    return None


# =========================================================
# NAME QUESTION
# =========================================================

def detect_name_question(words):

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

    return None


# =========================================================
# HOW QUESTION
# =========================================================

def detect_how_question(words):

    if (
        "how" in words
        and "you" in words
    ):

        return [
            "you",
            "how"
        ]

    return None


# =========================================================
# WHERE QUESTION
# =========================================================

def detect_where_question(words):

    if "where" not in words:
        return None

    content = [
        word
        for word in words
        if word not in {
            "where",
            "are",
            "is",
            "do",
            "does"
        }
    ]

    if content:
        return content + ["where"]

    return ["where"]


# =========================================================
# WHAT QUESTION
# =========================================================

def detect_what_question(words):

    if "what" not in words:
        return None

    if "your" in words and "name" in words:
        return None

    content = [
        word
        for word in words
        if word not in {
            "what",
            "is",
            "are",
            "do",
            "does"
        }
    ]

    if content:
        return content + ["what"]

    return ["what"]


# =========================================================
# WHY QUESTION
# =========================================================

def detect_why_question(words):

    if "why" not in words:
        return None

    content = [
        word
        for word in words
        if word not in {
            "why",
            "is",
            "are",
            "do",
            "does"
        }
    ]

    if content:
        return content + ["why"]

    return ["why"]


# =========================================================
# YES / NO QUESTION
# =========================================================

def detect_yes_no_question(words):

    if not words:
        return None

    question_starters = {
        "are",
        "is",
        "am",
        "do",
        "does",
        "did",
        "can",
        "will",
        "would",
        "have",
        "has"
    }

    if words[0] not in question_starters:
        return None

    # Remove auxiliary verb
    content = words[1:]

    # Remove English question structure
    content = [
        word
        for word in content
        if word not in {
            "you",
            "i",
            "he",
            "she",
            "they"
        }
    ]

    if not content:
        return None

    # Simple ASL-oriented structure
    if "you" in words:
        return ["you"] + content

    return content


# =========================================================
# NEGATION
# =========================================================

def detect_negation(words):

    negative_words = {
        "not",
        "dont",
        "don't",
        "never",
        "cannot",
        "can't",
        "no"
    }

    if not any(
        word in negative_words
        for word in words
    ):
        return None

    # Normalize common negative forms
    result = []

    for word in words:

        if word in {
            "dont",
            "don't",
            "cannot",
            "can't"
        }:
            continue

        result.append(word)

    # Put NOT at the end
    result.append("not")

    return result


# =========================================================
# REQUEST
# =========================================================

def detect_request(words):

    if "please" not in words:
        return None

    result = [
        word
        for word in words
        if word != "please"
    ]

    # Put object/topic before request
    if result:

        return result + ["please"]

    return ["please"]


# =========================================================
# TIME
# =========================================================

TIME_WORDS = {
    "today",
    "tomorrow",
    "yesterday",
    "morning",
    "night",
    "later",
    "now"
}


def move_time_to_front(words):

    time_words = []
    remaining = []

    for word in words:

        if word in TIME_WORDS:
            time_words.append(word)

        else:
            remaining.append(word)

    return time_words + remaining


# =========================================================
# GENERAL GLOSS
# =========================================================

def general_gloss(words):

    return move_time_to_front(
        words
    )


# =========================================================
# MAIN GLOSS ENGINE
# =========================================================

def create_gloss(text):

    words = clean_text(text)

    if not words:
        return []

    # ---------------------------------------------
    # Introduction
    # ---------------------------------------------

    result = detect_introduction(
        words
    )

    if result:
        return result

    # ---------------------------------------------
    # Name question
    # ---------------------------------------------

    result = detect_name_question(
        words
    )

    if result:
        return result

    # ---------------------------------------------
    # How question
    # ---------------------------------------------

    result = detect_how_question(
        words
    )

    if result:
        return result

    # ---------------------------------------------
    # Where question
    # ---------------------------------------------

    result = detect_where_question(
        words
    )

    if result:
        return result

    # ---------------------------------------------
    # What question
    # ---------------------------------------------

    result = detect_what_question(
        words
    )

    if result:
        return result

    # ---------------------------------------------
    # Why question
    # ---------------------------------------------

    result = detect_why_question(
        words
    )

    if result:
        return result

    # ---------------------------------------------
    # Negation
    # ---------------------------------------------

    result = detect_negation(
        words
    )

    if result:
        return result

    # ---------------------------------------------
    # Request
    # ---------------------------------------------

    result = detect_request(
        words
    )

    if result:
        return result

    # ---------------------------------------------
    # Yes / No question
    # ---------------------------------------------

    result = detect_yes_no_question(
        words
    )

    if result:
        return result

    # ---------------------------------------------
    # General sentence
    # ---------------------------------------------

    return general_gloss(
        words
    )


# =========================================================
# FINGERSPELLING
# =========================================================

def create_fingerspelling(word):

    return [
        character.upper()
        for character in word
        if character.isalpha()
    ]


# =========================================================
# NAME DETECTION
# =========================================================

def is_name(word, gloss, index):

    if index < 2:
        return False

    return (
        gloss[index - 2] == "my"
        and
        gloss[index - 1] == "name"
    )


# =========================================================
# CLASSIFY GLOSS
# =========================================================

def classify_gloss(gloss):

    sequence = []
    unsupported = []

    for index, word in enumerate(gloss):

        # -----------------------------------------
        # Name → fingerspell
        # -----------------------------------------

        if is_name(
            word,
            gloss,
            index
        ):

            sequence.append({
                "word": word,
                "type": "fingerspell",
                "letters": create_fingerspelling(
                    word
                )
            })

            continue

        # -----------------------------------------
        # Known sign
        # -----------------------------------------

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

        # -----------------------------------------
        # Unsupported
        # -----------------------------------------

        unsupported.append(
            word
        )

    return sequence, unsupported


# =========================================================
# COMPLETE PIPELINE
# =========================================================

def process_sentence(text):

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

def display_result(result):

    print("\n" + "=" * 70)
    print("SPEECH → ASL-ORIENTED SIGN PIPELINE")
    print("=" * 70)

    print("\nEnglish:")
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

        print("  No supported signs.")

    for item in result["sequence"]:

        if item["type"] == "sign":

            print(
                f"  🤟 {item['word']:<15}"
                f"SIGN"
            )

        elif item["type"] == "fingerspell":

            letters = " → ".join(
                item["letters"]
            )

            print(
                f"  🔤 {item['word']:<15}"
                f"FINGERSPELL → {letters}"
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

        print("  None")

    print("=" * 70)


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
