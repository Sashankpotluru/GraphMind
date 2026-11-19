"""Build the Pages artifact from verified records and an explicit asset list."""

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release import inventory, read, table
from benchmarks.sources import destination, digest, open_url, verified
from benchmarks.verify import ORIGINAL_RELEASE, verify
from graphmind import retrieve

CORE_FILES = ("__init__.py", "documents.py", "export.py", "graph.py", "retrieval.py")
WEB_FILES = (
    "index.html",
    "style.css",
    "app.mjs",
    "graph-view.mjs",
    "results-view.mjs",
    "data.mjs",
    "engine-client.mjs",
    "engine-worker.mjs",
    "parity.mjs",
    "dom.mjs",
    "assets/mark.svg",
    "assets/bricolage.woff2",
)
DOC_FILES = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "docs/architecture.svg",
    "docs/architecture.mmd",
    "docs/engine-contract.md",
    "docs/answer-protocol.md",
    "docs/retrieval-protocol.md",
    "benchmarks/sources.json",
    "benchmarks/model.json",
    "benchmarks/vendor/flan/LICENSE",
    "benchmarks/vendor/hotpot/LICENSE.txt",
    "benchmarks/vendor/musique/LICENSE",
    "benchmarks/vendor/scorers.json",
    "web/assets/Font-LICENSE.txt",
    "web/assets/Pyodide-LICENSE.txt",
    "web/assets/Python-LICENSE.txt",
)
RUNTIME = (
    "pyodide.mjs",
    "pyodide.asm.js",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
)
ARCHIVE = {
    "url": "https://registry.npmjs.org/pyodide/-/pyodide-0.29.3.tgz",
    "bytes": 5598203,
    "sha256": "924f3167d6bf5e553bda392615eaa82edd82a9cffc32a1cc49853a926b148c5c",
}


def write_json(target, data):
    target.write_text(
        json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf8",
    )


def allowed_file(base, name):
    path = base / name
    if (
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or path.resolve() != path.absolute()
        or not path.is_file()
    ):
        raise ValueError(f"unsafe or missing asset: {name}")
    return path


def copy_asset(source, target, row):
    path = allowed_file(source, row["path"])
    target_path = target / row["path"]
    target_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(verified(path, row), target_path)
    verified(target_path, row)


def bundle_core(target):
    with zipfile.ZipFile(target, "w") as archive:
        for name in CORE_FILES:
            info = zipfile.ZipInfo("graphmind/" + name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, allowed_file(ROOT / "src/graphmind", name).read_bytes())


def runtime(target):
    archive = ROOT / "data/cache/browser/pyodide-0.29.3.tgz"
    if not archive.exists():
        with (
            destination(archive, ARCHIVE) as temp,
            open_url(ARCHIVE["url"]) as src,
            temp.open("wb") as dest,
        ):
            shutil.copyfileobj(src, dest)
    target.mkdir()
    with tarfile.open(verified(archive, ARCHIVE), "r:gz") as entries:
        for name in RUNTIME:
            member = entries.getmember("package/" + name)
            if not member.isfile():
                raise ValueError(f"unexpected runtime member: {name}")
            with entries.extractfile(member) as src, (target / name).open("wb") as dest:
                shutil.copyfileobj(src, dest)


def sample_catalog(results):
    index = read(results / "explorer-index.json")["datasets"]
    catalog = {}
    real = None
    for dataset, ids in index.items():
        selected = set(ids["answer_ids"])
        dev = results / "retrieval" / dataset / "dev"
        cases = {c["qid"]: c for c in table(dev / "cases") if c["qid"] in selected}
        metrics = {m["qid"]: m for m in table(dev / "metrics") if m["qid"] in selected}
        catalog[dataset] = [
            {
                "qid": qid,
                "question": cases[qid]["question"],
                "delta": metrics[qid]["graph2_minus_hybrid_all4"],
            }
            for qid in ids["answer_ids"]
        ]
        if dataset == "hotpot":
            real = cases[ids["answer_ids"][0]]
    return catalog, real


def parity(name, case):
    payload = {"documents": case["documents"], "question": case["question"], "depth": 2}
    result = retrieve(**payload)
    return {
        "name": name,
        "payload": payload,
        "expected": {
            k: result[k] for k in ("ranking", "hops", "seed_weights", "per_seed", "graph")
        },
    }


def public_paths(release_files):
    return (
        set(WEB_FILES)
        | set(DOC_FILES)
        | {"runtime/" + name for name in RUNTIME}
        | {"results/" + row["path"] for row in release_files}
        | {".nojekyll", "engine.zip", "parity.json", "catalog.json", "results/release.json"}
    )


def site_inventory(target, expected):
    rows = inventory(target, ("site-manifest.json",))
    paths = {row["path"] for row in rows}
    if paths != expected:
        raise ValueError(f"site allowlist differs: {sorted(paths ^ expected)}")
    write_json(target / "site-manifest.json", {"schema": 1, "files": rows})
    if rows != inventory(target, ("site-manifest.json",)):
        raise ValueError("site changed during packaging")


def assemble(target):
    for source, names in [(ROOT / "web", WEB_FILES), (ROOT, DOC_FILES)]:
        for name in names:
            output = target / name
            output.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(allowed_file(source, name), output)
    results = ROOT / "results"
    manifest = results / "release.json"
    if digest(manifest) != ORIGINAL_RELEASE:
        raise ValueError("original release manifest changed")
    output = target / "results"
    output.mkdir()
    release_files = read(manifest)["files"]
    for row in release_files:
        copy_asset(results, output, row)
    shutil.copyfile(manifest, output / manifest.name)
    bundle_core(target / "engine.zip")
    runtime(target / "runtime")
    catalog, real = sample_catalog(output)
    write_json(target / "catalog.json", catalog)
    write_json(
        target / "parity.json",
        [
            parity("Authored bridge", read(ROOT / "tests/fixtures/bridge.json")),
            parity("HotpotQA sample 1", real),
        ],
    )
    (target / ".nojekyll").touch()
    site_inventory(target, public_paths(release_files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    target = ROOT / "site"
    verify(ROOT / "results")
    # Stage first. Interrupted builds leave the previous public artifact recoverable.
    with tempfile.TemporaryDirectory(dir=ROOT / "build") as folder:
        stage = Path(folder) / "site"
        stage.mkdir()
        assemble(stage)
        previous = Path(folder) / "previous"
        if target.exists():
            os.replace(target, previous)
        try:
            os.replace(stage, target)
        except OSError:
            if previous.exists():
                os.replace(previous, target)
            raise
    print("Built verified Pages artifact in site/.")


if __name__ == "__main__":
    (ROOT / "build").mkdir(exist_ok=True)
    main()
