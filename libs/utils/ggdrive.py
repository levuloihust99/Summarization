import os
import re
import json
import httpx
import argparse

from urllib.parse import urljoin
from urllib3.filepost import choose_boundary
from httpx import Response

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .file_utils import format_print_path, check_for_ignore


SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DRIVE_API_BASE_URL = "https://www.googleapis.com/upload/drive/v3/"
FILE_SIZE_THRESHOLD = 5 * 1024**2
client_secrets_file = "client_secrets.json"
token_file = "token.json"

# pattern search variables
include_pattern = None
ignore_pattern = None
include_over_ignore = True


def check_for_error(resp: Response):
    if resp.status_code >= 300:
        raise Exception(resp.content.decode())


def upload(file_path: str, parent_id: str):
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, SCOPES
            )
            creds = flow.run_local_server()
            with open(token_file, "w") as writer:
                writer.write(creds.to_json())
    authorization_headers = {"Authorization": "Bearer {}".format(creds.token)}
    service = build("drive", "v3", credentials=creds)

    print("Scanning...")
    file_path = os.path.abspath(file_path)
    stack = [file_path]
    sequence = []
    while stack:
        f = stack.pop()
        if check_for_ignore(
            f,
            ignore_pattern=ignore_pattern,
            include_pattern=include_pattern,
            include_over_ignore=include_over_ignore,
        ):
            continue
        sequence.append(f)
        if os.path.isdir(f):
            subpaths = os.listdir(f)
            subpaths = [os.path.join(f, p) for p in subpaths]
            subpaths = [p for p in subpaths if os.path.isfile(p) or os.path.isdir(p)]
            stack.extend(subpaths)

    print("Uploading...")
    id_tracker = {}
    count = len(sequence)
    max_num_width = len(str(count))
    counter_prefix_template = "#File {:0%d}/{}" % max_num_width
    drive_api_file_endpoint = urljoin(DRIVE_API_BASE_URL, "files")

    with httpx.Client() as client:
        for i, f in enumerate(sequence):
            counter_prefix = counter_prefix_template.format(i + 1, count)
            iterating_status = "{}: {}".format(
                counter_prefix, format_print_path(f, max_line_len=150)
            )
            print(iterating_status, end="")
            f_dir = os.path.dirname(f)
            _parent = id_tracker.get(f_dir, parent_id)
            if os.path.isfile(f):
                file_size = os.path.getsize(f)
                with open(f, "rb") as reader:
                    file_content = reader.read()
                file_metadata = {"name": os.path.basename(f), "parents": [_parent]}
                if file_size < FILE_SIZE_THRESHOLD:  # multipart upload
                    boundary = choose_boundary()
                    encoded_boundary = boundary.encode()
                    headers = {
                        "Content-Type": 'multipart/related; boundary="{}"'.format(
                            boundary
                        )
                    }
                    body = b""
                    body += b"--" + encoded_boundary + b"\n"
                    body += b"Content-Type: application/json; charset=UTF-8\n\n"
                    body += json.dumps(file_metadata).encode() + b"\n"
                    body += b"--" + encoded_boundary + b"\n"
                    body += b"Content-Type: application/octet-stream\n\n"
                    body += file_content + b"\n"
                    body += b"--" + encoded_boundary + b"--"
                    resp = client.post(
                        drive_api_file_endpoint,
                        content=body,
                        headers={**headers, **authorization_headers},
                    )
                    check_for_error(resp)
                else:  # use resumable upload
                    resp = client.post(
                        drive_api_file_endpoint,
                        params={"uploadType": "resumable"},
                        json=file_metadata,
                        headers={
                            "Content-Type": "application/json; charset=UTF-8",
                            **authorization_headers,
                        },
                    )
                    check_for_error(resp)
                    session_uri = resp.headers.get("location")
                    bytes_sent = 0
                    chunk_size = FILE_SIZE_THRESHOLD
                    while True:
                        chunk = file_content[bytes_sent : bytes_sent + chunk_size]
                        if len(chunk) == 0:
                            break
                        resp = client.put(
                            session_uri,
                            content=chunk,
                            headers={
                                **{
                                    "Content-Length": str(len(chunk)),
                                    "Content-Range": "bytes {}-{}/{}".format(
                                        bytes_sent,
                                        bytes_sent + len(chunk) - 1,
                                        file_size,
                                    ),
                                },
                                **authorization_headers,
                            },
                        )
                        if resp.status_code < 300:
                            break
                        elif resp.status_code == 308:
                            sent_range = resp.headers.get("range", "")
                            match = re.match(
                                r"^bytes=\d+-(?P<last_byte>\d+)$", sent_range
                            )
                            if not match:
                                raise Exception("Cannot determine bytes range")
                            bytes_sent = int(match.group("last_byte")) + 1
                            print("\r" + " " * len(iterating_status) + "\r", end="")
                            iterating_status = "{} ({}): {}".format(
                                counter_prefix,
                                "{}/{}".format(bytes_sent, file_size),
                                format_print_path(f),
                            )
                            print(iterating_status, end="")
                        else:
                            raise Exception(resp.content.decode())
            else:
                file_metadata = {
                    "name": os.path.basename(f),
                    "parents": [_parent],
                    "mimeType": "application/vnd.google-apps.folder",
                }
                uploaded_folder = service.files().create(body=file_metadata, fields="id").execute()
                id_tracker[f] = uploaded_folder.get("id")

            if i < len(sequence) - 1:
                print("\r" + " " * len(iterating_status) + "\r", end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        "-f",
        required=True,
        help="Path to the file or directory that you want to upload.",
    )
    parser.add_argument(
        "--client_secrets_file",
        "-c",
        default="client_secrets.json",
        help="Path to the credentials file, containing the client_id and client_secret.",
    )
    parser.add_argument(
        "--token_file",
        "-t",
        default="token.json",
        help="Path to the token file, containing the access_token and the refresh_token",
    )
    parser.add_argument(
        "--parent_id",
        default="root",
        help="ID of the directory that will contain the file to upload.",
    )
    parser.add_argument(
        "--ignore_pattern",
        nargs="+",
        default=None,
        help="Exclude files matching this pattern.",
    )
    parser.add_argument(
        "--include_pattern",
        nargs="+",
        default=None,
        help="Include files matching this pattern.",
    )
    parser.add_argument(
        "--include_over_ignore",
        default=True,
        type=eval,
        help="If True, files not matching `ignore_pattern` or matching `include_pattern` will be included. "
        "Else, files matching `include_pattern` and not matching `ignore_pattern` will be included.",
    )
    args = parser.parse_args()

    global include_pattern, ignore_pattern, include_over_ignore, client_secrets_file, token_file
    include_pattern = args.include_pattern
    ignore_pattern = args.ignore_pattern
    include_over_ignore = args.include_over_ignore

    upload(args.file_path, args.parent_id)


if __name__ == "__main__":
    main()
