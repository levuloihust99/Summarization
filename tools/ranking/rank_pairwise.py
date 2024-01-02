import os
import re
import json
import asyncio
import logging
import argparse
import motor.motor_asyncio

from tqdm import tqdm
from pathlib import Path
from decouple import config
from typing import Text, List, Dict, Optional

from libs.utils.logging import add_color_formater
from libs.utils.mongo_utils import setup_db
from ..generative import (
    prompting,
    AVAILABLE_MODELS,
    get_api_keys
)

logging.basicConfig(level=logging.DEBUG)
add_color_formater(logging.root)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Your job is to select between two summaries of an Vietnamese news article one that is better. You should consider several criterion such as accuracy, coverage and coherence during evaluating each summary. Each summary has an ID. You must reply with and only with the ID of the preferred summary. Here are the article and the two summaries:

Article:
\"\"\"
{article}
\"\"\"

Summaries:
---------------
[ID={id1}] {summary1}
---------------
[ID={id2}] {summary2}
"""

HEURISTIC_ORDER = [
    ("gpt-3.5-turbo", "Open-Orca/Mistral-7B-OpenOrca"),
    ("Open-Orca/Mistral-7B-OpenOrca", "vinai/bartpho-syllalble_finetune-step100"),
    ("gpt-3.5-turbo", "vinai/bartpho-syllalble_finetune-step100"),
    ("vinai/bartpho-syllalble_finetune-step100", "Reference"),
    ("Open-Orca/Mistral-7B-OpenOrca", "Reference"),
    ("gpt-3.5-turbo", "Reference"),
    ("NlpHUST/t5-small-vi-summarization", "Reference"),
    ("NlpHUST/t5-small-vi-summarization", "vinai/bartpho-syllalble_finetune-step100"),
    ("NlpHUST/t5-small-vi-summarization", "Open-Orca/Mistral-7B-OpenOrca"),
    ("NlpHUST/t5-small-vi-summarization", "gpt-3.5-turbo")
]


class Node:
    def __init__(
        self,
        name: Optional[Text] = "default",
        children: Optional[Dict[Text, "Node"]] = None,
        parent: Optional["Node"] = None
    ):
        self.name = name
        self.children = children or {}
        self.parent = parent

    def __repr__(self):
        return "Node[{}]".format(self.name)

    def is_smaller(self, other: "Node"):
        node = self.parent
        while node:
            if node is other:
                return True
            node = node.parent
        return False


def infer_rank(node: Node):
    rank = [node.name]
    while True:
        if len(node.children) > 1:
            raise Exception(
                "Cannot infer rank due to multiple children per node: node {} has {:d} children {}"
                .format(node.name, len(node.children), list(node.children.keys()))
            )
        elif len(node.children) == 0:
            break
        child_key = next(iter(node.children.keys()))
        node = node.children[child_key]
        rank.append(node.name)
    return rank


async def rank_pairwise(sampleId: Text, article: Text, summaries: List[Dict]):
    loop = asyncio.get_running_loop()
    args = loop.ctx.args

    rank_path = os.path.join(args.storage_dir, sampleId, "rank.json")
    if os.path.exists(rank_path):
        return

    summary_mapping = {
        summary["metadata"]["generator"]: summary
        for summary in summaries
    }
    generators = list(summary_mapping.keys())

    def parse(llm_output: Text) -> Text:
        matches = re.finditer(r"{}".format("|".join(generators)), llm_output)
        matches = list(matches)
        if len(matches) != 1:
            logger.warning("Cannot determine preferred summary. Leave for manually selection.")
            return None
        return matches[0].group()

    node_mapping = {"root": Node(name="root")}
    comparisons = {}
    completed_path = os.path.join(args.storage_dir, sampleId, "comparisons.json")
    if os.path.exists(completed_path):
        with open(completed_path, 'r') as reader:
            comparisons = json.load(reader)
        comparisons = {
            eval(k): v for k, v in comparisons.items()
        }

    for idx, (x, y) in enumerate(HEURISTIC_ORDER):
        if x not in node_mapping:
            node_x = Node(name=x, parent=node_mapping["root"])
            node_mapping["root"].children[x] = node_x
            node_mapping[x] = node_x
        else:
            node_x = node_mapping[x]
        if y not in node_mapping:
            node_y = Node(name=y, parent=node_mapping["root"])
            node_mapping["root"].children[y] = node_y
            node_mapping[y] = node_y
        else:
            node_y = node_mapping[y]

        if (x, y) in comparisons:
            comparison = comparisons[(x, y)]
            preferred = comparison.get("preferred", None)
            if preferred is None:
                logger.warning("Cannot determine preferred summary for sampleId {}, ({}, {})".format(sampleId, x, y))
                continue
            else:
                if preferred == 1:
                    preferred_generator = x
                    other_generator = y
                else:
                    preferred_generator = y
                    other_generator = x
                node_mapping[preferred_generator].children[other_generator] = node_mapping[other_generator]
                node_mapping[other_generator].parent.children.pop(other_generator)
                node_mapping[other_generator].parent = node_mapping[preferred_generator]

        if node_x.is_smaller(node_y) or node_y.is_smaller(node_x):
            continue

        compare_prompt = PROMPT_TEMPLATE.format(
            article=article,
            id1=x, id2=y,
            summary1=summary_mapping[x]["content"],
            summary2=summary_mapping[y]["content"]
        )

        kwargs = {}
        if args.model_name.startswith("gpt-3.5"):
            kwargs["api_key"] = loop.ctx.get_api_key()
        elif args.model_name.startswith("gpt-4"):
            kwargs["api_key"] = loop.ctx.gpt4_api_key
        else: # gemini
            kwargs["candidate_count"] = 1
            kwargs["api_key"] = loop.ctx.get_gemini_api_key()
        output = await prompting(
            prompt=compare_prompt,
            model=args.model_name,
            **kwargs
        )
        preferred_generator = parse(output)
        if preferred_generator is None:
            logger.warning("sampleId {}, compare ({}, {})".format(sampleId, x, y))
            continue
        assert preferred_generator in [x, y]

        other_generator = x if x != preferred_generator else y
        node_mapping[preferred_generator].children[other_generator] = node_mapping[other_generator]
        node_mapping[other_generator].parent.children.pop(other_generator)
        node_mapping[other_generator].parent = node_mapping[preferred_generator]

        preferred = 1 if x == preferred_generator else -1
        comparisons[(x, y)] = {
            "preferred": preferred,
            "model": args.model_name,
            "completion": output,
            "prompt": compare_prompt,
        }
        sample_dir = os.path.join(args.storage_dir, sampleId)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        with open(os.path.join(sample_dir, "comparisons.json"), "w") as writer:
            serialized_comparisons = {
                str(k): v for k, v in comparisons.items()
            }
            json.dump(serialized_comparisons, writer, indent=4, ensure_ascii=False)

    try:
        rank = infer_rank(node_mapping["root"])[1:]
        sample_dir = os.path.join(args.storage_dir, sampleId)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        rank_file = os.path.join(sample_dir, "rank.json")
        with open(rank_file, "w") as writer:
            json.dump(rank, writer, indent=4, ensure_ascii=False)
    except:
        pass


async def process_doc(doc: Dict):
    loop = asyncio.get_running_loop()
    args = loop.ctx.args
    await rank_pairwise(sampleId=doc["sampleId"], article=doc["input"], summaries=doc["outputs"])


async def launch(args):
    loop = asyncio.get_running_loop()
    setattr(loop, "ctx", argparse.Namespace())

    if args.model_name.startswith("gpt-3.5"):
        assert args.gpt35_api_key_file
        api_keys = get_api_keys(args.gpt35_api_key_file)
        idx = -1
        def get_api_key():
            nonlocal idx
            idx = (idx + 1) % len(api_keys)
            return api_keys[idx]
        loop.ctx.get_api_key = get_api_key
    elif args.model_name.startswith("gpt-4"):
        assert args.gpt4_api_key
        loop.ctx.gpt4_api_key = args.gpt4_api_key
    else: # gemini-pro
        assert args.gemini_api_key_file
        api_keys = get_api_keys(args.gemini_api_key_file)
        idx = -1
        def get_api_key():
            nonlocal idx
            idx = (idx + 1) % len(api_keys)
            return api_keys[idx]
        loop.ctx.get_gemini_api_key = get_api_key

    mongo_client = setup_db(
        host=config('MONGO_HOST'),
        port=config('MONGO_PORT'),
        username=config('MONGO_USERNAME'),
        password=config('MONGO_PASSWORD')
    )
    loop.ctx.mongo_client = mongo_client
    loop.ctx.mongo_collection = mongo_client[args.mongo_schema][args.mongo_collection]
    loop.ctx.args = args

    storage_dir = os.path.join(args.storage_dir, args.model_name)
    args.storage_dir = storage_dir
    if not os.path.exists(args.storage_dir):
        os.makedirs(args.storage_dir)

    cursor = loop.ctx.mongo_collection.find()
    progress = tqdm()
    batch = []
    async for doc in cursor:
        batch.append(process_doc(doc))
        if len(batch) == args.concurrent_factor:
            await asyncio.gather(*batch)
            progress.update(len(batch))
            batch = []

    if batch:
        await asyncio.gather(*batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongo_schema", default="rank-reward")
    parser.add_argument("--mongo_collection", default="ranking-samples-filtered")
    parser.add_argument("--storage_dir", default=Path(__file__).parent / "data/pairwise_ranking" / "v1")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", choices=AVAILABLE_MODELS)
    parser.add_argument("--gpt4_api_key", default=config('GPT4_API_KEY', default=None)),
    parser.add_argument("--gpt35_api_key_file", default=config('OPENAI_API_KEY_FILE', default=None))
    parser.add_argument("--gemini_api_key_file", default=config('GEMINI_API_KEY_FILE', default=None))
    parser.add_argument("--concurrent_factor", type=int, default=1)
    args = parser.parse_args()

    asyncio.run(launch(args))


if __name__ == "__main__":
    main()
