import re


# ============================================================
# SIGNAFLOW AI - SENTENCE PROCESSOR
# English → simple ASL-oriented gloss
# ============================================================


REMOVE_WORDS = {
    "a",
    "an",
    "the",
    "am",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "and",
    "will",
    "can",
    "do",
    "did",
}


# ============================================================
# WORDS WITH ACTUAL SIGN VIDEO ASSETS
# ============================================================

NORMAL_SIGN_WORDS = {

    "hello",
    "morning",
    "night",
    "afternoon",
    "good",

    "i",
    "you",
    "my",
    "your",
    "name",
    "me",

    "go",
    "come",
    "school",
    "college",
    "home",

    "water",
    "food",
    "breakfast",
    "lunch",
    "dinner",

    "please",
    "yes",
    "no",

    "drink",
    "eat",

    "happy",
    "sad",
    "fine",
    "better",
    "bad",

    "have",
    "want",
    "need",
    "give",
    "help",

    "what",
    "where",
    "when",
    "who",
    "why",
    "how",

    "today",
    "tomorrow",
    "yesterday",
    "now",
    "later",

    "listen",
    "see",
    "hear",
    "talk",
    "read",
    "write",
    "say",
    "understand",

    "love",
    "like",
    "know",
    "study",
    "work",

    "mother",
    "father",
    "boy",
    "girl",
    "man",
    "woman",

    "book",
    "open",
    "friend",
    "family",

    "wait",
    "same",
    "close",

    "every",
    "first",
    "sleep",

    "outside",
    "out",
    "here",
    "stay",

    "hungry",
    "thirsty",

    "take",
}


# ============================================================
# NATURAL WORD ALIASES
# ============================================================

WORD_ALIASES = {

    "hi": "hello",

    "mom": "mother",

    "dad": "father",

    "thanks": "thankyou",

    "thank": "thankyou",
}


# ============================================================
# COMMON PHRASES
# ============================================================

PHRASE_MAP = {

    # --------------------------------------------------------
    # Greetings
    # --------------------------------------------------------

    "hi": [
        "HELLO"
    ],

    "hello": [
        "HELLO"
    ],

    "good morning": [
        "GOOD",
        "MORNING"
    ],

    "good night": [
        "GOOD",
        "NIGHT"
    ],

    "good afternoon": [
        "GOOD",
        "AFTERNOON"
    ],


    # --------------------------------------------------------
    # Family
    # --------------------------------------------------------

    "hello mom": [
        "HELLO",
        "MOTHER"
    ],

    "hi mom": [
        "HELLO",
        "MOTHER"
    ],

    "hello dad": [
        "HELLO",
        "FATHER"
    ],

    "hi dad": [
        "HELLO",
        "FATHER"
    ],


    # --------------------------------------------------------
    # Food / drink
    # --------------------------------------------------------

    "i am hungry": [
        "I",
        "HUNGRY"
    ],

    "i am thirsty": [
        "I",
        "THIRSTY"
    ],

    "i want food": [
        "I",
        "WANT",
        "FOOD"
    ],

    "i want water": [
        "I",
        "WANT",
        "WATER"
    ],

    "give me water": [
        "GIVE",
        "ME",
        "WATER"
    ],

    "give me food": [
        "GIVE",
        "ME",
        "FOOD"
    ],

    "please give water": [
        "PLEASE",
        "GIVE",
        "WATER"
    ],

    "give water please": [
        "PLEASE",
        "GIVE",
        "WATER"
    ],

    "drink water": [
        "DRINK",
        "WATER"
    ],

    "eat food": [
        "EAT",
        "FOOD"
    ],

    "water please": [
        "WATER",
        "PLEASE"
    ],

    "breakfast": [
        "BREAKFAST"
    ],

    "lunch": [
        "LUNCH"
    ],

    "dinner": [
        "DINNER"
    ],

    "i had breakfast": [
        "I",
        "BREAKFAST"
    ],

    "i want breakfast": [
        "I",
        "WANT",
        "BREAKFAST"
    ],

    "i want lunch": [
        "I",
        "WANT",
        "LUNCH"
    ],

    "i want dinner": [
        "I",
        "WANT",
        "DINNER"
    ],


    # --------------------------------------------------------
    # Movement
    # --------------------------------------------------------

    "go home": [
        "GO",
        "HOME"
    ],

    "i am going home": [
        "I",
        "GO",
        "HOME"
    ],

    "i am going to school": [
        "I",
        "GO",
        "SCHOOL"
    ],

    "i am going to college": [
        "I",
        "GO",
        "COLLEGE"
    ],

    "go outside": [
        "GO",
        "OUTSIDE"
    ],

    "i want to go outside": [
        "I",
        "WANT",
        "GO",
        "OUTSIDE"
    ],

    "come home": [
        "COME",
        "HOME"
    ],

    "wait please": [
        "PLEASE",
        "WAIT"
    ],

    "please wait": [
        "PLEASE",
        "WAIT"
    ],


    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    "i will go tomorrow": [
        "I",
        "GO",
        "TOMORROW"
    ],

    "see you tomorrow": [
        "SEE",
        "YOU",
        "TOMORROW"
    ],

    "see you later": [
        "SEE",
        "YOU",
        "LATER"
    ],

    "today i am happy": [
        "TODAY",
        "I",
        "HAPPY"
    ],

    "today i am sad": [
        "TODAY",
        "I",
        "SAD"
    ],


    # --------------------------------------------------------
    # Feelings
    # --------------------------------------------------------

    "i am happy": [
        "I",
        "HAPPY"
    ],

    "i am sad": [
        "I",
        "SAD"
    ],

    "i am fine": [
        "I",
        "FINE"
    ],


    # --------------------------------------------------------
    # Questions
    # --------------------------------------------------------

    "what is your name": [
        "YOUR",
        "NAME",
        "WHAT"
    ],

    "where are you": [
        "YOU",
        "WHERE"
    ],

    "how are you": [
        "YOU",
        "HOW"
    ],

    "what are you doing": [
        "YOU",
        "WHAT"
    ],

    "where is my home": [
        "MY",
        "HOME",
        "WHERE"
    ],


    # --------------------------------------------------------
    # Common
    # --------------------------------------------------------

    "thank you": [
        "THANKYOU"
    ],

    "thanks": [
        "THANKYOU"
    ],

    "please help me": [
        "PLEASE",
        "HELP",
        "ME"
    ],

    "help me": [
        "HELP",
        "ME"
    ],

    "i need help": [
        "I",
        "NEED",
        "HELP"
    ],
}


# ============================================================
# FINGERSPELLING
# ============================================================

def is_fingerspellable_word(word):

    return bool(
        re.fullmatch(
            r"[a-zA-Z]+",
            word
        )
    )


def fingerspell_word(word):

    if not is_fingerspellable_word(word):
        return []

    return list(
        word.upper()
    )


# ============================================================
# WORD → SIGN TOKEN(S)
# ============================================================

def word_to_signs(word):

    word = (
        word
        .lower()
        .strip()
    )

    if not word:
        return []

    word = WORD_ALIASES.get(
        word,
        word
    )

    if word == "thankyou":

        return [
            "THANKYOU"
        ]

    if word in NORMAL_SIGN_WORDS:

        return [
            word.upper()
        ]

    return fingerspell_word(
        word
    )


# ============================================================
# ENGLISH → ASL-STYLE GLOSS
# ============================================================

def translate_to_asl_gloss(text):

    text = (
        text or ""
    ).lower().strip()

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

    if not text:
        return []


    # --------------------------------------------------------
    # Exact common phrase
    # --------------------------------------------------------

    if text in PHRASE_MAP:

        return PHRASE_MAP[
            text
        ]


    words = text.split()


    # --------------------------------------------------------
    # SELF INTRODUCTION
    # --------------------------------------------------------

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

        result = [
            "HELLO",
            "MY",
            "NAME",
        ]

        result.extend(
            fingerspell_word(
                name
            )
        )

        return result


    # --------------------------------------------------------
    # MY NAME IS...
    # --------------------------------------------------------

    if (
        len(words) >= 4
        and words[0] == "my"
        and words[1] == "name"
        and words[2] == "is"
    ):

        name = words[3]

        result = [
            "MY",
            "NAME",
        ]

        result.extend(
            fingerspell_word(
                name
            )
        )

        return result


    # --------------------------------------------------------
    # NAME QUESTION
    # --------------------------------------------------------

    if (
        "what" in words
        and "your" in words
        and "name" in words
    ):

        return [
            "YOUR",
            "NAME",
            "WHAT",
        ]


    # --------------------------------------------------------
    # BASIC QUESTIONS
    # --------------------------------------------------------

    if (
        "how" in words
        and "you" in words
    ):

        return [
            "YOU",
            "HOW",
        ]


    if (
        "where" in words
        and "you" in words
    ):

        return [
            "YOU",
            "WHERE",
        ]


    # --------------------------------------------------------
    # GENERAL PROCESSING
    # --------------------------------------------------------

    result = []

    time_words = {
        "today",
        "tomorrow",
        "yesterday",
        "morning",
        "afternoon",
        "night",
        "later",
    }

    time_tokens = []

    other_tokens = []


    for word in words:

        if word in REMOVE_WORDS:
            continue

        word = WORD_ALIASES.get(
            word,
            word
        )

        if word in time_words:

            time_tokens.append(
                word.upper()
            )

        else:

            other_tokens.extend(
                word_to_signs(
                    word
                )
            )


    result.extend(
        time_tokens
    )

    result.extend(
        other_tokens
    )

    return result


# ============================================================
# DISPLAY HELPER
# ============================================================

def gloss_to_text(gloss):

    return " ".join(
        str(word)
        for word in gloss
    )
