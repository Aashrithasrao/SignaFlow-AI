import re
import speech_recognition as sr


# =========================================================
# SIGN LANGUAGE VOCABULARY
# =========================================================

SIGN_VOCABULARY = {
    "bad",
    "better",
    "book",
    "boy",
    "close",
    "dad",
    "drink",
    "every",
    "fine",
    "first",
    "food",
    "girl",
    "give",
    "go",
    "happy",
    "have",
    "hear",
    "hello",
    "home",
    "hungry",
    "later",
    "listen",
    "look",
    "man",
    "mom",
    "morning",
    "night",
    "no",
    "now",
    "open",
    "outside",
    "please",
    "read",
    "sad",
    "same",
    "say",
    "see",
    "sleep",
    "stay",
    "talk",
    "thankyou",
    "thirsty",
    "tomorrow",
    "wait",
    "wake",
    "water",
    "where",
    "who",
    "why",
    "yes",
}


# =========================================================
# COMMON PHRASES
# =========================================================

PHRASE_MAP = {
    "thank you": "thankyou",
    "good morning": "morning",
    "good night": "night",
}


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):
    """
    Convert speech-recognition output into
    normalized lowercase words.
    """

    text = text.lower()

    # Replace common phrases first
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

    return text.split()


# =========================================================
# FINGERSPELLING
# =========================================================

def create_fingerspell_sequence(word):
    """
    Convert a name such as 'ashrita'
    into individual fingerspelling letters.
    """

    letters = []

    for letter in word:

        if letter.isalpha():

            letters.append({
                "letter": letter.upper(),
                "type": "fingerspell",
                "asset": (
                    "signs/fingerspelling/"
                    f"{letter.upper()}.mp4"
                )
            })

    return letters


# =========================================================
# DETECT PERSON'S NAME
# =========================================================

def is_likely_name(word, previous_words):
    """
    Detect the name in a phrase such as:

        my name is ashrita

    The word immediately after:
        my name is

    is treated as a possible name.
    """

    if len(previous_words) >= 3:

        last_three = previous_words[-3:]

        if last_three == [
            "my",
            "name",
            "is"
        ]:

            return True

    return False


# =========================================================
# CREATE SIGN SEQUENCE
# =========================================================

def create_sign_sequence(text):

    words = clean_text(text)

    sequence = []
    unsupported = []

    previous_words = []

    for word in words:

        # -------------------------------------------------
        # Normal sign
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

        # -------------------------------------------------
        # Possible person's name
        # -------------------------------------------------

        elif is_likely_name(
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

        # -------------------------------------------------
        # Unsupported word
        # -------------------------------------------------

        else:

            unsupported.append(word)

        previous_words.append(word)

    return {
        "original_text": text,
        "sequence": sequence,
        "unsupported": unsupported
    }


# =========================================================
# PRINT RESULT
# =========================================================

def print_result(result):

    print("\n" + "=" * 60)
    print("TEXT → SIGN ANALYSIS")
    print("=" * 60)

    print("\nOriginal:")
    print(result["original_text"])

    print("\nSign sequence:")

    if not result["sequence"]:

        print("  No supported signs.")

    else:

        for item in result["sequence"]:

            # Normal sign
            if item["type"] == "sign":

                print(
                    f"  {item['word']} "
                    f"→ SIGN "
                    f"→ {item['asset']}"
                )

            # Fingerspelling
            elif item["type"] == "fingerspell":

                letters = " → ".join(
                    letter["letter"]
                    for letter in item["letters"]
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


# =========================================================
# SPEECH RECOGNITION
# =========================================================

def listen_once():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        print("=" * 60)
        print("SPEECH → SIGN")
        print("=" * 60)

        print(
            "\nAdjusting for background noise..."
        )

        recognizer.adjust_for_ambient_noise(
            source,
            duration=1
        )

        print("🎤 Speak now...")

        try:

            audio = recognizer.listen(
                source,
                timeout=10,
                phrase_time_limit=10
            )

        except sr.WaitTimeoutError:

            print(
                "\nNo speech detected."
            )

            return None

    print(
        "\nProcessing speech..."
    )

    try:

        text = recognizer.recognize_google(
            audio
        )

        return text

    except sr.UnknownValueError:

        print(
            "\nSorry, I couldn't understand "
            "the speech."
        )

    except sr.RequestError as error:

        print(
            "\nSpeech recognition service "
            f"error: {error}"
        )

    return None


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    text = listen_once()

    if text:

        print("\nYou said:")
        print(text)

        result = create_sign_sequence(
            text
        )

        print_result(
            result
        )
