import json
from sign_sequence import process_sentence


text = input(
    "Enter sentence: "
)


result = process_sentence(
    text
)


with open(
    "dataset/sign_sequence.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        result,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\nSign sequence saved!")

print(
    "dataset/sign_sequence.json"
)
