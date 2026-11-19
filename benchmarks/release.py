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


def explorer(root):
    """IDs are in stable coordinate order; each table's shard size is explicit."""
    from graphmind import build_graph

    retrieval = root / "retrieval"
    selected = read(root / "answers/selection.json")
    index = {
        "schema": 1,
        "retrieval_shard_size": 64,
        "answer_shard_size": 64,
        "graph_shard_size": 8,
        "methods": list(ANSWER_METHODS),
        "datasets": {},
    }
    graphs = {}
    for ds in COUNTS:
        cases = canonical_rows(retrieval, ds, "dev", "cases")
        by_id = {c["qid"]: c for c in cases}
        ids = selection(by_id, "graphmind-v1-answer:", 100)
        require_equal(ds + " explorer membership", ids, selected[ds])
        index["datasets"][ds] = {"dev_ids": list(by_id), "answer_ids": ids}
        graphs[ds] = [{"qid": q, "graph": build_graph(by_id[q]["documents"])} for q in ids]
    return index, graphs


def build(observations, output):
    import shutil
    import tempfile

    from .run_retrieval import compressed, put_json
    from .verify import verify

    if output.exists() or observations.resolve() in output.resolve().parents:
        raise ValueError("release output must be new and outside the observations directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ("." + output.name + ".lock")
    with lock.open("x"):
        try:
            with tempfile.TemporaryDirectory(
                dir=output.parent, prefix="." + output.name + "-"
            ) as temp:
                stage = Path(temp) / "release"
                stage.mkdir()
                for name in ("retrieval", "answers"):
                    if any(p.is_symlink() for p in (observations / name).rglob("*")):
                        raise ValueError("symlinks in observations")
                    shutil.copytree(observations / name, stage / name)
                sidecar = observations / "retrieval-provenance.json"
                if sidecar.exists():
                    shutil.copyfile(sidecar, stage / sidecar.name)
                index, graphs = explorer(stage)
                put_json(stage / "explorer-index.json", index)
                for ds, rows in graphs.items():
                    for i in range(0, len(rows), 8):
                        compressed(
                            stage / "explorer" / ds / f"{i // 8:04}.jsonl.gz",
                            b"".join(json_bytes(r) + b"\n" for r in rows[i : i + 8]),
                        )
                put_json(stage / "release.json", {"schema": 1, "files": inventory(stage)})
                verify(stage, recorded=sidecar.exists())
                if output.exists():
                    raise FileExistsError(output)
                stage.rename(output)
        finally:
            lock.unlink(missing_ok=True)


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Build and verify a new release from completed observations."
    )
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    opts = p.parse_args()
    build(opts.observations, opts.output)


if __name__ == "__main__":
    main()
