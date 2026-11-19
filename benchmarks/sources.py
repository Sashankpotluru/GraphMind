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


def open_url(url):
    """Use verified HTTPS even on Python distributions installed without a CA bundle."""
    import ssl

    context = ssl.create_default_context()
    if not context.get_ca_certs():
        import certifi

        context.load_verify_locations(certifi.where())
    return urllib.request.urlopen(url, timeout=120, context=context)


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
                        open_url(spec["url"]) as src,
                        temp.open("wb") as dest,
                    ):
                        shutil.copyfileobj(src, dest)


# Commit binding covers both algorithm and measurement inputs. Other user work
# in the checkout, such as unrelated documentation, need not be pristine.
MEASUREMENT_PATHS = (
    "benchmarks",
    "src/graphmind",
    "scripts/download_data.py",
    "requirements-benchmark.txt",
    "pyproject.toml",
    "docs/retrieval-protocol.md",
    "docs/answer-protocol.md",
)


def source_binding(checkout):
    import subprocess

    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args])

    commit = git("rev-parse", "--verify", "HEAD").decode().strip()
    if git("status", "--porcelain=v1", "--untracked-files=all", "--", *MEASUREMENT_PATHS):
        raise ValueError(
            "dirty measurement or engine paths; commit measurement source before running"
        )
    names = git("ls-tree", "-r", "-z", "--name-only", commit, "--", *MEASUREMENT_PATHS)
    hashes = {}
    for raw in names.split(b"\0"):
        if not raw:
            continue
        name = raw.decode()
        target = checkout / name
        expected = hashlib.sha256(git("show", commit + ":" + name)).hexdigest()
        if target.is_symlink() or digest(target) != expected:
            raise ValueError(f"measurement file differs from Git source: {name}")
        hashes[name] = expected
    return {"commit": commit, "source_sha256": hashes}
