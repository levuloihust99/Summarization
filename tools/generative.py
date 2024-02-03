import json
import aiohttp
import logging
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

import openai
import google.generativeai as genai
gemini = genai.GenerativeModel("gemini-pro")

from typing import Text, List, Optional, Literal

from libs.restful.utils import parse_aiohttp_error


AVAILABLE_MODELS = [
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-0301",
    "gemini-pro",
]


class GeminiException(Exception):
    def __init__(self, *args, **kwargs):
        super(GeminiException, self).__init__(*args)
        self.kwargs = kwargs


class NetworkGeminiException(Exception):
    def __init__(self, *, error_msg: Text, error_class: Text):
        self.error_msg = error_msg
        self.error_class = error_class


def get_api_keys(api_key_file: Text) -> List[Text]:
    api_keys = []
    with open(api_key_file, "r") as reader:
        for line in reader:
            line = line.strip()
            if line:
                api_keys.append(line)
    return api_keys


async def openai_chat_generate(prompt: Text, **kwargs):
    """Call OpenAI API to generate output from a prompt (for ChatCompletion endpoint).
    
    Valid kwargs are:
        model (Text): name of the OpenAI model, e.g. gpt-3.5-turbo, gpt-4-1106-preview, .etc
        api_key (Text): OpenAI API key that authenticates the request.
        temperature (float): control the randomness of the output.
    
    Return (Text):
        The completion of the prompt.
    """

    response = await openai.ChatCompletion.acreate(
        messages=[
            {"role": "user", "content": prompt}
        ],
        **kwargs
    )
    output = response.choices[0].message.content
    return output


async def openai_text_generate(prompt: Text, **kwargs):
    """Call OpenAI API to generate output from a prompt (for Completion endpoint).
    
    Valid kwargs are:
        model (Text): name of the OpenAI model, e.g. gpt-3.5-turbo-instruct, .etc
        api_key (Text): OpenAI API key that authenticates the request.
        temperature (float): control the randomness of the output.
    
    Return (Text):
        The completion of the prompt.
    """

    response = await openai.Completion.acreate(
        prompt=prompt,
        **kwargs
    )
    output = response.choices[0].text
    return output


async def gemini_generate(prompt: Text, **kwargs):
    """Call Gemini API to generate output from a prompt.
    
    Valid kwargs are:
        api_key
        candidate_count
        stop_sequences
        max_output_tokens
        temperature
        top_p
        top_k
    
    Return (Text):
        The completion of the prompt.
    """
    api_key = kwargs.pop("api_key")
    genai.configure(api_key=api_key)
    gen_config = genai.types.GenerationConfig(**kwargs)
    try:
        response = await gemini.generate_content_async(
            prompt,
            generation_config=gen_config,
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        )
        return response.text
    except Exception as e:
        logger.error(e)
        raise GeminiException(wrapped=e)


async def network_gemini_generate(url: Text, prompt: Text, **kwargs):
    """Call a worker that is in charge of calling Gemini API.

    Args:
        url (Text): url of the worker
        prompt (Text): the input prompt

    Valid kwargs are:
        api_key (required)
        candidate_count
        stop_sequences
        max_output_tokens
        temperature
        top_p
        top_k
        Return (Text):
        The completion of the prompt.
    """

    data = {
        "api_key": kwargs.pop("api_key"),
        "prompt": prompt,
        "kwargs": kwargs
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"}) as response:
            error = await parse_aiohttp_error(response)
            if error is not None:
                if isinstance(error, dict):
                    error_msg = error.get("error")
                    error_class = error.get("error_class")
                else:
                    error_msg = error
                    error_class = None
                logger.error(error_msg)
                raise NetworkGeminiException(error_msg=error_msg, error_class=error_class)
            resp_data = await response.json()
            return resp_data["completion"]


async def prompting(prompt: Text, model: Text, **kwargs):
    if model == "gemini-pro":
        return await gemini_generate(prompt, **kwargs)
    elif model.startswith("gpt"):
        if "instruct" in model:
            return await openai_text_generate(prompt, model=model, **kwargs)
        else:
            return await openai_chat_generate(prompt, model=model, **kwargs)
    else:
        raise Exception("Model '{}' is not supported".format(model))


class GenWorkerPicker:
    GEMINI_ENDPOINT = "/gemini-pro"
    OPENAI_ENDPOINT = "/openai"

    def __init__(
        self,
        worker_pool_file: Text,
        openai_api_key_pool_file: Optional[Text] = None,
        gemini_api_key_pool_file: Optional[Text] = None
    ):
        """
        Args:
            worker_pool_file (Text): path to the file containing workers' info.
            openai_api_key_pool_file (Text): path to the file containing api keys for OpenAI.
            gemini_api_key_pool_file (Text): path to the file containing api keys for Gemini.
        """
        worker_pool = []
        with open(worker_pool_file, "r") as reader:
            for line in reader:
                line = line.strip()
                if line:
                    worker_pool.append(line)
        if not worker_pool:
            raise Exception("Empty worker pool.")
        self.worker_pool = worker_pool

        self.openai_api_key_pool = None
        self.openai_worker_idx = -1
        self.openai_worker_api_key_idx_tracker = {}
        if openai_api_key_pool_file:
            openai_api_key_pool = []
            with open(openai_api_key_pool_file, "r") as reader:
                for line in reader:
                    line = line.strip()
                    if line:
                        openai_api_key_pool.append(line)
            if not openai_api_key_pool:
                raise Exception("Empty OpenAI api key pool.")
            self.openai_api_key_pool = openai_api_key_pool
            for worker in self.worker_pool:
                self.openai_worker_api_key_idx_tracker[worker] = -1

        self.gemini_api_key_pool = None
        self.gemini_worker_idx = -1
        self.gemini_worker_api_key_idx_tracker = {}
        if gemini_api_key_pool_file:
            gemini_api_key_pool = []
            with open(gemini_api_key_pool_file, "r") as reader:
                for line in reader:
                    line = line.strip()
                    if line:
                        gemini_api_key_pool.append(line)
            if not gemini_api_key_pool:
                raise Exception("Empty Gemini api key pool.")
            self.gemini_api_key_pool = gemini_api_key_pool
            for worker in self.worker_pool:
                self.gemini_worker_api_key_idx_tracker[worker] = -1

    def pick(self, model_name: Literal['gemini-pro', 'openai']):
        if model_name == "gemini-pro":
            self.gemini_worker_idx = (self.gemini_worker_idx + 1) % len(self.worker_pool)
            gemini_worker = self.worker_pool[self.gemini_worker_idx]
            self.gemini_worker_api_key_idx_tracker[gemini_worker] = \
                (self.gemini_worker_api_key_idx_tracker[gemini_worker] + 1) % len(self.gemini_api_key_pool)
            api_key = self.gemini_api_key_pool[self.gemini_worker_api_key_idx_tracker[gemini_worker]]
            return urljoin(gemini_worker, self.GEMINI_ENDPOINT), api_key
        else:
            self.openai_worker_idx = (self.openai_worker_idx + 1) % len(self.worker_pool)
            openai_worker = self.worker_pool[self.openai_worker_idx]
            self.openai_worker_api_key_idx_tracker[openai_worker] = \
                (self.openai_worker_api_key_idx_tracker[openai_worker] + 1) % len(self.openai_api_key_pool)
            api_key = self.openai_api_key_pool[self.openai_worker_api_key_idx_tracker[openai_worker]]
            return urljoin(openai_worker, self.OPENAI_ENDPOINT), api_key
