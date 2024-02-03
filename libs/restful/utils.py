import json
import aiohttp

from typing import Text


async def parse_aiohttp_error(response: aiohttp.ClientResponse) -> Text:
    if response.status != 200:
        try:
            response_text = ""
            response_text = await response.text()
            error_msg = json.loads(response_text)
        except:
            error_msg = response_text or "Unknown error"
        return error_msg
    return None
