"""Read and bind experiment artifacts without loading neural weights."""

import gzip
import hashlib
import json
from pathlib import Path

from .data import COUNTS, selection
from .generation import ANSWER_METHODS, VENDOR, build_prompt
from .run_retrieval import json_bytes
from .sources import digest

CANONICAL = json.loads((VENDOR / "canonical.json").read_text())


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf8") as f:
            return json.load(f)
    return json.loads(path.read_text())


def table(folder):
    for path in sorted(folder.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf8") as f:
            for line in f:
                yield json.loads(line)


def inventory(root, exclude=()):
    rows = []
    for p in sorted(root.rglob("*")):
        name = p.relative_to(root).as_posix()
        if p.is_symlink() or not (p.is_file() or p.is_dir()):
            raise ValueError(f"non-regular artifact: {name}")
        if p.is_file() and name not in exclude:
            rows.append({"path": name, "bytes": p.stat().st_size, "sha256": digest(p)})
    return rows


def require_equal(label, expected, actual):
    if expected != actual:
        raise ValueError(f"mismatch: {label}")


def check_inventory(root, expected, exclude=()):
    require_equal("artifact inventory", expected, inventory(root, exclude))


def bound_rows(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(json_bytes(r) + b"\n")
    return h.hexdigest()


def canonical_rows(base, ds, split, name):
    rows = list(table(base / ds / split / name))
    require_equal(
        f"{ds}/{split}/{name} source binding", CANONICAL[ds][split][name], bound_rows(rows)
    )
    return rows


def answer_inputs(retrieval, tok):
    ids, inputs = {}, {}
    for ds in COUNTS:
        cases = canonical_rows(retrieval, ds, "dev", "cases")
        selected = ids[ds] = selection((c["qid"] for c in cases), "graphmind-v1-answer:", 100)
        wanted = set(selected)
        by_id = {c["qid"]: c for c in cases if c["qid"] in wanted}
        runs = {r["qid"]: r for r in table(retrieval / ds / "dev/runs") if r["qid"] in wanted}
        inputs[ds] = []
        for q in selected:
            c = by_id[q]
            for method in ANSWER_METHODS:
                docs = [c["documents"][i] for i in runs[q]["rankings"][method][:4]]
                prompt = build_prompt(tok, c["question"], docs)
                inputs[ds].append({"qid": q, "method": method, **prompt})
    return ids, inputs


def judgments(retrieval):
    return {
        ds: {r["qid"]: r for r in canonical_rows(retrieval, ds, "dev", "judgments")}
        for ds in COUNTS
    }


def original_sources(checkout, commit, paths):
    import subprocess

    return {
        p: hashlib.sha256(
            subprocess.check_output(["git", "-C", str(checkout), "show", f"{commit}:{p}"])
        ).hexdigest()
        for p in paths
    }


def verify_source_files():
    for item in read(VENDOR / "scorers.json"):
        name = item["source_path"]
        content = (VENDOR / item["dataset"] / Path(name)).read_bytes()
        # These two import-only adaptations keep the upstream metric functions verbatim.
        if name == "hotpot_evaluate_v1.py":
            content = content.replace(b"import json\n", b"import ujson as json\n", 1)
        elif name == "metrics/answer.py":
            content = content.replace(
                b"from .metric import Metric", b"from metrics.metric import Metric", 1
            )
        require_equal(name, item["sha256"], hashlib.sha256(content).hexdigest())
