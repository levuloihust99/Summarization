import urllib.parse
import motor.motor_asyncio
from typing import Text, Union, Optional


def setup_db(
    host: Text,
    port: Union[int, Text],
    username: Optional[Text] = None,
    password: Optional[Text] = None,
):
    if username and password:
        db_auth = "{}:{}@".format(
            urllib.parse.quote_plus(username),
            urllib.parse.quote_plus(password)
        )
    else:
        db_auth = ""
    connection_uri = "mongodb://{}{}:{}".format(db_auth, host, port)
    db_client = motor.motor_asyncio.AsyncIOMotorClient(connection_uri)
    return db_client
