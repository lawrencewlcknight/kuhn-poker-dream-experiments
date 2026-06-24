"""Upload an output directory to Google Cloud Storage from a Batch VM.

This avoids invoking the Google Cloud CLI inside the Batch task. The CLI Python
runtime can diverge from the experiment Python runtime on managed Batch images,
which makes post-run uploads a surprisingly fragile final step.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


METADATA_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/default/token"
)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Destination must start with gs://, got: {uri}")

    path = uri[len("gs://") :]
    bucket, sep, prefix = path.partition("/")
    if not bucket:
        raise ValueError(f"Destination is missing a bucket name: {uri}")

    return bucket, prefix.strip("/")


def iter_files(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            yield path


def get_metadata_access_token() -> str:
    request = Request(METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
    with urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))

    token = payload.get("access_token")
    if not token:
        raise RuntimeError("Metadata server response did not include access_token")
    return token


def upload_file(bucket: str, object_name: str, path: Path, token: str) -> None:
    encoded_bucket = quote(bucket, safe="")
    encoded_name = quote(object_name, safe="")
    url = (
        f"https://storage.googleapis.com/upload/storage/v1/b/{encoded_bucket}/o"
        f"?uploadType=media&name={encoded_name}"
    )
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = path.read_bytes()
    request = Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
            "Content-Length": str(len(data)),
        },
    )

    with urlopen(request, timeout=120) as response:
        response.read()


def upload_with_retries(bucket: str, object_name: str, path: Path, token: str) -> None:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            upload_file(bucket, object_name, path, token)
            return
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {details}")
        except URLError as exc:
            last_error = exc

        if attempt < 3:
            time.sleep(2**attempt)

    raise RuntimeError(f"Failed to upload {path} to gs://{bucket}/{object_name}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument(
        "destination",
        help="GCS destination prefix, for example gs://bucket/job-name/outputs",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    if not source_dir.exists():
        print(f"No output directory found at {source_dir}; nothing to upload.")
        return 0
    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")

    bucket, prefix = parse_gcs_uri(args.destination)
    files = list(iter_files(source_dir))
    if not files:
        print(f"No files found under {source_dir}; nothing to upload.")
        return 0

    token = get_metadata_access_token()
    uploaded_bytes = 0
    for path in files:
        relative_path = path.relative_to(source_dir).as_posix()
        object_name = f"{prefix}/{relative_path}" if prefix else relative_path
        upload_with_retries(bucket, object_name, path, token)
        uploaded_bytes += path.stat().st_size
        print(f"Uploaded {path} to gs://{bucket}/{object_name}")

    print(f"Uploaded {len(files)} files ({uploaded_bytes} bytes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
