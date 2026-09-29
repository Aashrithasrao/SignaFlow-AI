from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SIGNS_DIR = PROJECT_ROOT / "signs"
LETTERS_DIR = SIGNS_DIR / "letters"


def get_word_video(word: str):
    """
    Return the video path for a known sign word.
    Returns None if the word does not have a dedicated video.
    """

    word = word.strip().lower()

    video = SIGNS_DIR / word / f"{word}.mp4"

    if video.exists() and video.stat().st_size > 0:
        return video

    return None


def get_letter_image(letter: str):
    """
    Return the fingerspelling image for a letter.
    """

    letter = letter.strip().upper()

    image = LETTERS_DIR / f"{letter}.png"

    if image.exists() and image.stat().st_size > 0:
        return image

    return None


def build_asset_sequence(words):
    """
    Convert a list of words into visual assets.

    Known words:
        word -> MP4

    Unknown words:
        word -> letter-by-letter fingerspelling
    """

    sequence = []

    for word in words:

        clean_word = word.strip().lower()

        if not clean_word:
            continue

        video = get_word_video(clean_word)

        if video is not None:

            sequence.append({
                "text": clean_word.upper(),
                "type": "video",
                "path": str(video),
            })

        else:

            letters = []

            for letter in clean_word:

                if not letter.isalpha():
                    continue

                image = get_letter_image(letter)

                if image is not None:
                    letters.append({
                        "text": letter.upper(),
                        "type": "fingerspell",
                        "path": str(image),
                    })

            if letters:

                sequence.extend(letters)

            else:

                sequence.append({
                    "text": clean_word.upper(),
                    "type": "missing",
                    "path": None,
                })

    return sequence
