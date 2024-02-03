import re
import json
import string
import asyncio
import unicodedata
import motor.motor_asyncio

from pathlib import Path
from typing import Text
from decouple import config

from text_tools.vietnamese_words.trie import Trie

vietnamese_word_path = config('VIETNAMESE_WORD_PATH')
trie = Trie()
with open(vietnamese_word_path, "r") as reader:
    for line in reader:
        trie.add(line.strip())

MONGO_URL = config('MONGO_URL')
filtered_path = Path(__file__).parent / "data" / "filtered_data.jsonl"


def strip_punctuation(text: Text) -> Text:
    pattern = r"[{}]".format(re.escape(string.punctuation))
    stripped_text = re.sub(pattern, " ", text)
    return stripped_text


not_vi = 0
empty = 0
prefix = 0
phrase = 0

def should_include(text: Text) -> bool:
    global not_vi, empty, prefix, phrase
    text = unicodedata.normalize("NFKC", text)
    text = strip_punctuation(text)
    text = text.lower()

    if text.startswith("bài báo") or text.startswith("bài viết") or text.startswith("văn bản"):
        prefix += 1
        return False

    words = text.split()
    prefix_words = words[:8]
    if "tóm tắt" in " ".join(prefix_words):
        phrase += 1
        return False

    if not words:
        empty += 1
        return False

    count = 0
    for word in words:
        if trie.exists(word):
            count += 1
    if count / len(words) <= 0.4:
        not_vi += 1
        return False

    return True


def filter_item(doc) -> bool:
    global ignore
    for output in doc["outputs"]:
        if should_include(output["content"]) is False:
            return True
    return False


async def launch():
    db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
    collection = db_client["rank-reward"]["ranking-samples"]

    with open(filtered_path, "w") as writer:
        cursor = collection.find({})
        async for doc in cursor:
            if filter_item(doc):
                continue
            doc["_id"] = str(doc["_id"])
            doc["createdAt"] = str(doc["createdAt"])
            doc["updatedAt"] = str(doc["updatedAt"])
            writer.write(json.dumps(doc, ensure_ascii=False) + "\n")
            writer.flush()


def main():
    asyncio.run(launch())
    print("Not vi: {}".format(not_vi))
    print("Empty: {}".format(empty))
    print("Prefix: {}".format(prefix))
    print("Phrase: {}".format(phrase))


if __name__ == "__main__":
    main()
