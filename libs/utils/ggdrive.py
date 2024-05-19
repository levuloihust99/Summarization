import os
import argparse

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from .file_utils import format_print_path, check_for_ignore


google_auth = GoogleAuth()
drive_app = GoogleDrive(google_auth)

# pattern search variables
include_pattern = None
ignore_pattern = None
include_over_ignore = True


def specify_config_file(file_path: str):
    google_auth.settings["client_config_file"] = file_path


def upload(file_path: str, parent_id: str):
    file_path = os.path.abspath(file_path)

    print("Scanning...")
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
    for i, f in enumerate(sequence):
        counter_prefix = counter_prefix_template.format(i + 1, count)
        iterating_status = "{}: {}".format(
            counter_prefix, format_print_path(f, max_line_len=150)
        )
        iterating_status_width = len(iterating_status)
        print(iterating_status, end="")
        f_dir = os.path.dirname(f)
        _parent = id_tracker.get(f_dir, parent_id)
        if os.path.isfile(f):
            file = drive_app.CreateFile(
                {"title": os.path.basename(f), "parents": [{"id": _parent}]}
            )
            file.SetContentFile(f)
            file.Upload()
        else:  # is directory
            folder = drive_app.CreateFile(
                {
                    "title": os.path.basename(f),
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [{"id": _parent}],
                }
            )
            folder.Upload()
            id_tracker[f] = folder["id"]
        if i < len(sequence) - 1:
            print("\r" + " " * iterating_status_width + "\r", end="")
    print()


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
        help="Path to the credentials file.",
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

    global include_pattern, ignore_pattern, include_over_ignore
    include_pattern = args.include_pattern
    ignore_pattern = args.ignore_pattern
    include_over_ignore = args.include_over_ignore

    specify_config_file(args.client_secrets_file)
    upload(args.file_path, parent_id=args.parent_id)


if __name__ == "__main__":
    main()
