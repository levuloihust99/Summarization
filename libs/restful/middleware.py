import logging
from datetime import datetime
from functools import partial
from typing import Optional

from sanic import Sanic

logger = logging.getLogger(__name__)


async def before_request_func(request):
    request.ctx.start_time = datetime.now()


async def after_response_func(request, response, service: Optional[str] = None):
    if service is not None:
        prefix = "[{} service] ".format(service)
    logger.info(
        "{}Total processing time: {}".format(
            prefix, datetime.now() - request.ctx.start_time
        )
    )


def register_middleware(app: Sanic, service: Optional[str] = None):
    app.register_middleware(before_request_func, "request")
    app.register_middleware(partial(after_response_func, service=service), "response")