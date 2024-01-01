import openai
import google.generativeai as genai
gemini = genai.GenerativeModel("gemini-pro")

from typing import Text, List


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


def get_api_keys(api_key_file: Text) -> List[Text]:
    api_keys = []
    with open(api_key_file, "r") as reader:
        for line in reader:
            line = line.strip()
            if line:
                api_keys.append(line)
    return api_keys


async def openai_generate(prompt: Text, **kwargs):
    """Call OpenAI API to generate output from a prompt.
    
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


async def gemini_generate(prompt: Text, **kwargs):
    """Call Gemini API to generate output from a prompt.
    
    Valid kwargs are:
    
    Return (Text):
        The completion of the prompt.
    """
    api_key = kwargs.pop("api_key")
    genai.configure(api_key=api_key)
    gen_config = genai.types.GenerationConfig(**kwargs)
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


async def prompting(prompt: Text, model: Text, **kwargs):
    if model == "gemini-pro":
        return await gemini_generate(prompt, **kwargs)
    elif model.startswith("gpt"):
        return await openai_generate(prompt, model=model, **kwargs)
    else:
        raise Exception("Model '{}' is not supported".format(model))