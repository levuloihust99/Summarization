import os
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

CONCURRENT_REQUESTS = 10
LIMIT = float("inf")
AVAILABLE_MODELS = [
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-0301",
    "gemini-pro",
]
STORAGE_PROMPT_FILE = "prompt.txt"


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
    for idx, summary in enumerate(summaries):
        content = summary["content"]
        generator = summary["metadata"]["generator"]
        formatted_summary = "[ID={}] {}".format(generator, content)
        formatted_summaries.append(formatted_summary)
    formatted_summaries = "\n--------------\n".join(formatted_summaries)
    ranking_prompt = prompt_template.format(
        article=doc["input"],
        summaries=formatted_summaries
    )
    return ranking_prompt


async def prompting(prompt: Text, model: Text) -> Text:
    if model == "gemini-pro":
        return await gemini_generate(prompt)
    elif model.startswith("gpt-4"):
        return await gpt4_generate(prompt, model)
    elif model.startswith("gpt-3.5"):
        return await gpt_35_generate(prompt, model)
    else:
        raise Exception("Model '{}' is not supported".format(model))


async def gpt_35_generate(prompt: Text, model_name) -> Text:
    api_key = get_openai_api_key()
    response = await openai.ChatCompletion.acreate(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        api_key=api_key
    )
    output = response.choices[0].message.content
    return output


async def gpt4_generate(prompt: Text, model_name: Text) -> Text:
    response = await openai.ChatCompletion.acreate(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        api_key=config("GPT4_API_KEY"),
        temperature=0.0
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
    loop = asyncio.get_running_loop()
    args = loop.ctx.args

    if not os.path.exists(args.storage_dir):
        os.makedirs(args.storage_dir)

    sample_ids = os.listdir(args.storage_dir)
    exist_data = []
    for sample_id in sample_ids:
        sample_id_path = os.path.join(args.storage_dir, sample_id)
        file_names = os.listdir(sample_id_path)
        model_completions = []
        prompt = None
        for file_name in file_names:
            model_name, ext = os.path.splitext(file_name)
            if model_name in AVAILABLE_MODELS:
                with open(os.path.join(sample_id_path, file_name), "r") as reader:
                    model_completion = reader.read()
                model_completions.append({"model": model_name, "completion": model_completion})
            elif model_name == "prompt":
                # assert model_name == "prompt"
                with open(os.path.join(sample_id_path, STORAGE_PROMPT_FILE), "r") as reader:
                    prompt = reader.read()
        exist_data.append({
            "sample_id": sample_id,
            "prompt": prompt,
            "completions": model_completions
        })

    return exist_data


async def make_request(doc):
    loop = asyncio.get_running_loop()
    args = loop.ctx.args

    ranking_prompt = get_ranking_prompt(doc)
    prompt_storage_path = os.path.join(args.storage_dir, doc["sampleId"], STORAGE_PROMPT_FILE)
    prompt_storage_dir = os.path.dirname(prompt_storage_path)
    if not os.path.exists(prompt_storage_dir):
        os.makedirs(prompt_storage_dir)
    with open(prompt_storage_path, "w") as writer:
        writer.write(ranking_prompt)

    output = await prompting(ranking_prompt, model=args.model_name)
    completion_storage_path = os.path.join(args.storage_dir, doc["sampleId"], f"{args.model_name}.txt")
    with open(completion_storage_path, "w") as writer:
        writer.write(output)


async def launch(args):
    loop = asyncio.get_running_loop()
    setattr(loop, "ctx", argparse.Namespace())
    loop.ctx.args = args

    db_client = setup_db()
    collection = db_client["rank-reward"]["ranking-samples-filtered"]

    exist_data = load_exist_data()
    exist_ids = set([item["sample_id"] for item in exist_data])

    def should_process(_doc):
        if _doc["sampleId"] not in exist_ids:
            return True
        sample_storage_dir = os.path.join(args.storage_dir, _doc["sampleId"])
        model_names = os.listdir(sample_storage_dir)
        for _model in model_names:
            _model = os.path.splitext(_model)[0]
            # assert _model in AVAILABLE_MODELS + ["prompt"]
            if _model == args.model_name:
                return False
        return True

    cursor = collection.find({})
    batch_requests = []
    if 0 < LIMIT < float("inf"):
        progress = tqdm(total=LIMIT, desc="Doc")
    else:
        progress = tqdm(desc="Doc")

    count = -1
    async for doc in cursor:
        if not should_process(doc):
            progress.update(1)
            continue

        count += 1
        if count < 0:
            progress.update(1)
            continue
        if 0 < LIMIT < float("inf") and count >= LIMIT:
            break

        exist_ids.add(doc["sampleId"])

        batch_requests.append(make_request(doc))
        if len(batch_requests) == CONCURRENT_REQUESTS:
            await asyncio.gather(*batch_requests)
            progress.update(len(batch_requests))
            batch_requests = []

    if batch_requests:
        await asyncio.gather(*batch_requests)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_dir", default=Path(__file__).parent / "v3")
    parser.add_argument("--model_name", required=True,
                        choices=AVAILABLE_MODELS)
    args = parser.parse_args()
    asyncio.run(launch(args))
    # with open("_tmp/rewrite_prompt.txt", "r") as reader:
    #     input_prompt = reader.read()
#     prompt = """I want you to rewrite the following prompt to enhance its clarity and grammatical correctness. The prompt is enclosed in \"\"\"\"\"\"

# \"\"\"
# {prompt}
# \"\"\"""".format(prompt=input_prompt)
    # prompt = input_prompt
    # output_prompt = asyncio.run(prompting(prompt, 'gpt-4-1106-preview'))
    # with open("_tmp/rewrite_prompt_output.txt", "w") as writer:
    #     writer.write(output_prompt)


if __name__ == "__main__":
    main()
