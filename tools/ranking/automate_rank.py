import re
import json
import openai
import asyncio
import logging
import argparse
import urllib.parse
import motor.motor_asyncio

from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone
from typing import Text, List
from decouple import config

from libs.utils.logging import add_color_formater

logging.basicConfig(level=logging.DEBUG)
add_color_formater(logging.root)

CONCURRENT_REQUESTS = 5
LIMIT = 20


def setup_db():
    username = config("MONGO_USERNAME")
    password = config("MONGO_PASSWORD")
    if username and password:
        db_auth = "{}:{}@".format(
            urllib.parse.quote_plus(username),
            urllib.parse.quote_plus(password)
        )
    else:
        db_auth = ""
    connection_uri = "mongodb://{}{}:{}".format(db_auth, config("MONGO_HOST"), config("MONGO_PORT"))
    db_client = motor.motor_asyncio.AsyncIOMotorClient(connection_uri)
    return db_client


def load_api_keys():
    api_key_file = config('OPENAI_API_KEY_FILE')
    api_keys = []
    with open(api_key_file, "r") as reader:
        for line in reader:
            api_keys.append(line.strip())
    return api_keys

api_keys = load_api_keys()
    

with open(Path(__file__).parent / "prompt_template.txt", "r") as reader:
    prompt_template = reader.read()


_idx = 0
def get_openai_api_key():
    global _idx, api_keys
    api_key = api_keys[_idx]
    _idx += 1
    return api_key


def get_ranking_prompt(doc):
    summaries = doc["outputs"]
    formatted_summaries = []
    for summary in summaries:
        content = summary["content"]
        generator = summary["metadata"]["generator"]
        formatted_summary = "[{}] {}".format(generator, content)
        formatted_summaries.append(formatted_summary)
    formatted_summaries = "\n--------------\n".join(formatted_summaries)
    ranking_prompt = prompt_template.format(
        article=doc["input"],
        summaries=formatted_summaries
    )
    return ranking_prompt


async def openai_generate(prompt: Text) -> Text:
    api_key = get_openai_api_key()
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        api_key=api_key
    )
    output = response.choices[0].message.content
    return output


async def gpt4_generate(prompt: Text) -> Text:
    response = await openai.ChatCompletion.acreate(
        model="gpt-4-0314",
        messages=[
            {"role": "user", "content": prompt}
        ],
        api_key=config("GPT4_API_KEY")
    )
    output = response.choices[0].message.content
    return output


async def gemini_generate(prompt: Text) -> Text:
    pass


def parse_output(ranking_output: Text) -> List[Text]:
    match = re.match(r"Ranking:\s*\[(.*?)\]", ranking_output)
    if not match:
        return None
    rankings = match.group(1).split(",")
    return rankings


def load_exist_data():
    ranking_results_file = Path(__file__).parent / "rankings.jsonl"
    if not ranking_results_file.exists():
        return []
    exist_data = []
    with open(ranking_results_file, "r") as reader:
        for line in reader:
            exist_data.append(json.loads(line.strip()))
    return exist_data


async def make_request(doc):
    ranking_prompt = get_ranking_prompt(doc)
    output = await openai_generate(ranking_prompt)
    rankings = parse_output(output)
    ranking_labeled_item = {
        "sampleId": doc["sampleId"],
        "prompt": ranking_prompt,
        "completion": output,
        "rankings": rankings
    }
    loop = asyncio.get_running_loop()
    writer = loop.ctx.file_writer
    writer.write(json.dumps(ranking_labeled_item, ensure_ascii=False) + "\n")
    writer.flush()


async def launch():
    db_client = setup_db()
    collection = db_client["rank-reward"]["ranking-samples-filtered"]

    exist_data = load_exist_data()
    exist_ids = set([item["sampleId"] for item in exist_data])
    file_writer = open(Path(__file__).parent / "rankings.jsonl", "a")
    loop = asyncio.get_running_loop()
    setattr(loop, "ctx", argparse.Namespace())
    loop.ctx.file_writer = file_writer

    cursor = collection.find({})
    batch_requests = []
    progress = tqdm()
    count = 0
    async for doc in cursor:
        if doc["sampleId"] in exist_ids:
            continue
        else:
            exist_ids.add(doc["sampleId"])
        count += 1
        if 0 < LIMIT < float("inf") and count >= LIMIT:
            break

        batch_requests.append(make_request(doc))
        if len(batch_requests) == CONCURRENT_REQUESTS:
            await asyncio.gather(*batch_requests)
            progress.update(len(batch_requests))
            batch_requests = []

    if batch_requests:
        await asyncio.gather(*batch_requests)

    file_writer.close()


def main():
    asyncio.run(launch())


if __name__ == "__main__":
    main()
