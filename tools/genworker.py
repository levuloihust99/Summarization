import logging
import argparse

import sanic
from sanic import Sanic
from sanic_cors import CORS
from sanic.request.types import Request

from libs.restful.middleware import register_middleware
from libs.utils.logging import do_setup_logging

from tools.generative import gemini_generate, GeminiException


do_setup_logging()
logger = logging.getLogger(__name__)

app = Sanic(__name__)
CORS(app)
register_middleware(app)


@app.route("/gemini-pro", methods=["POST"])
async def gemini_pro_handler(request: Request):
    logger.info("POST /gemini-pro")
    data = request.json

    api_key = data.get("api_key")
    if not api_key:
        error_msg = "Missing required parameter '{}'".format("api_key")
        logger.error(error_msg)
        return sanic.json({"error": error_msg}, status=400)

    prompt = data.get("prompt")
    if not prompt:
        error_msg = "Missing required parameter '{}'".format("prompt")
        logger.error(error_msg)
        return sanic.json({"error": error_msg}, status=400)

    gen_kwargs = data.get("gen_kwargs", {})
    try:
        output = await gemini_generate(prompt=prompt, api_key=api_key, **gen_kwargs)
        return sanic.json({"completion": output})
    except GeminiException as e:
        return sanic.json({"error": str(e.kwargs["wrapped"]), "error_class": e.kwargs["wrapped"].__class__.__qualname__}, status=500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7377)
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port, single_process=True)


if __name__ == "__main__":
    main()
