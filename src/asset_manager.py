from pathlib import Path


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

SIGNS_DIR = (
    PROJECT_ROOT / "signs"
)


# =========================================================
# FIND SIGN ASSET
# =========================================================

def find_sign(word):

    word = word.lower().strip()

    possible_files = [

        SIGNS_DIR
        / word
        / f"{word}.mp4",

        SIGNS_DIR
        / word
        / f"{word}.webm",

        SIGNS_DIR
        / word
        / f"{word}.gif",

        SIGNS_DIR
        / f"{word}.mp4",

        SIGNS_DIR
        / f"{word}.webm",

        SIGNS_DIR
        / f"{word}.gif",
    ]

    for file in possible_files:

        if file.exists():

            return {
                "word": word,
                "type": "sign",
                "found": True,
                "path": str(file)
            }

    return {
        "word": word,
        "type": "unsupported",
        "found": False,
        "path": None
    }


# =========================================================
# FINGERSPELL
# =========================================================

def fingerspell(word):

    letters = []

    for character in word:

        if character.isalpha():

            letters.append(
                character.upper()
            )

    return {
        "word": word,
        "type": "fingerspell",
        "found": True,
        "letters": letters
    }


# =========================================================
# CHECK WORD
# =========================================================

def classify_word(word):

    word = word.lower().strip()

    # ---------------------------------------------
    # First try actual sign asset
    # ---------------------------------------------

    sign = find_sign(
        word
    )

    if sign["found"]:

        return sign

    # ---------------------------------------------
    # Single letter
    # ---------------------------------------------

    if len(word) == 1 and word.isalpha():

        return fingerspell(
            word
        )

    # ---------------------------------------------
    # Otherwise unsupported
    # ---------------------------------------------

    return {
        "word": word,
        "type": "unsupported",
        "found": False,
        "path": None
    }


# =========================================================
# CHECK SENTENCE
# =========================================================

def check_words(words):

    results = []

    for word in words:

        results.append(
            classify_word(
                word
            )
        )

    return results


# =========================================================
# DISPLAY
# =========================================================

def display_results(results):

    print("\n" + "=" * 60)
    print("SIGN ASSET MANAGER")
    print("=" * 60)

    for result in results:

        word = result["word"]

        if result["type"] == "sign":

            print(
                f"🤟 {word:<15}"
                f"SIGN"
            )

            print(
                f"   Asset: "
                f"{result['path']}"
            )

        elif (
            result["type"]
            == "fingerspell"
        ):

            letters = " → ".join(
                result["letters"]
            )

            print(
                f"🔤 {word:<15}"
                f"FINGERSPELL"
            )

            print(
                f"   {letters}"
            )

        else:

            print(
                f"⚠️ {word:<15}"
                f"UNSUPPORTED"
            )

    print("=" * 60)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "SIGN ASSET MANAGER"
    )

    print(
        "=" * 60
    )

    words = input(
        "\nEnter words separated by spaces: "
    ).lower().split()

    results = check_words(
        words
    )

    display_results(
        results
    )
