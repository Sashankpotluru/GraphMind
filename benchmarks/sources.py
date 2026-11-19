"""Fetch immutable inputs, checking bytes before publication or model loading."""

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "sources.json").read_text())
MODEL = json.loads((HERE / "model.json").read_text())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verified(path, spec):
    if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
        raise ValueError(f"size/digest differs for {path.name}")
    return path


@contextmanager
def destination(path, spec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temp = Path(f.name)
    try:
        yield temp
        verified(temp, spec)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def extract_member(archive, spec, path):
    name = spec["archive_member"]
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or "\\" in name:
        raise ValueError("unsafe archive member")
    with zipfile.ZipFile(archive) as z:
        matches = [i for i in z.infolist() if i.filename == name]
        if len(matches) != 1 or matches[0].file_size != spec["bytes"]:
            raise ValueError("unexpected member size or count")
        with z.open(matches[0]) as src, destination(path, spec) as temp, temp.open("wb") as dest:
            shutil.copyfileobj(src, dest)


def fetch(cache):
    for spec in sorted(DATA + MODEL, key=lambda row: "archive_member" in row):
        path = cache / Path(spec["path"]).relative_to("data/cache")
        if path.exists():
            verified(path, spec)
        elif "archive_member" in spec:
            extract_member(cache / "musique_v1.0.zip", spec, path)
        else:
            with destination(path, spec) as temp:
                if "drive.google.com" in spec["url"]:
                    import gdown

                    if (
                        gdown.download(id="1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h", output=str(temp))
                        is None
                    ):
                        raise ValueError("archive download failed")
                else:
                    with (
                        urllib.request.urlopen(spec["url"], timeout=120) as src,
                        temp.open("wb") as dest,
                    ):
                        shutil.copyfileobj(src, dest)
