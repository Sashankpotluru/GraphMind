"""Reconstruct the release without network access, source datasets or neural weights."""

import argparse
import gzip
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np

from graphmind import bm25, build_graph

from .data import COUNTS, membership, selection
from .generation import FLAN, SETTINGS, analyze, check_output, tokenizer
from .models import WIDTH, model_identity
from .release import (
    CANONICAL,
    answer_inputs,
    bound_rows,
    canonical_rows,
    check_inventory,
    explorer,
    judgments,
    original_sources,
    read,
    require_equal,
    table,
    verify_source_files,
)
from .run_retrieval import METHODS, json_bytes, run_split, tune
from .sources import DATA, HERE, MEASUREMENT_PATHS, digest
from .statistics import components, holm, summarize

ORIGINAL_RELEASE = "4fb8dea4e49520bbc7d80230252ec584fedc4290d11a60d80c59ca3fcaf53740"
RETRIEVAL_MANIFEST = "a8b84048a64817155acf3f0697bce3c2df8cb31f293a54a54069c4017dd457e8"
SIDECAR = "b0d5ea6152966f64ef463c9340972cd77494017d3cbf6ac2c01922a094384668"
CHECKOUT = HERE.parent


def git_binding(commit, hashes):
    import subprocess

    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("invalid source revision")
    names = subprocess.check_output(
        [
            "git",
            "-C",
            str(CHECKOUT),
            "ls-tree",
            "-r",
            "-z",
            "--name-only",
            commit,
            "--",
            *MEASUREMENT_PATHS,
        ]
    )
    paths = [p.decode() for p in names.split(b"\0") if p]
    require_equal(f"{commit} source hashes", original_sources(CHECKOUT, commit, paths), hashes)


def checked_sources(root, recorded):
    verify_source_files()
    retrieval = root / "retrieval"
    manifest = read(retrieval / "manifest.json")
    require_equal("retrieval manifest schema", set(manifest), {"schema", "files"})
    check_inventory(retrieval, manifest["files"], ("manifest.json",))
    provenance = read(retrieval / "provenance.json")
    model, identity = model_identity()
    require_equal("retrieval model", model, provenance["model"])
    require_equal("encoder identity", identity, provenance["encoder_identity"])
    require_equal("source files", DATA, provenance["datasets"])
    manifest_hash = digest(retrieval / "manifest.json")
    if recorded:
        require_equal("original manifest", RETRIEVAL_MANIFEST, manifest_hash)
        require_equal("post-run sidecar", SIDECAR, digest(root / "retrieval-provenance.json"))
        sidecar = read(root / "retrieval-provenance.json")
        require_equal("historical qualification", "post-run", sidecar["assembly"])
        require_equal("historical manifest", manifest_hash, sidecar["manifest_sha256"])
        hashes = sidecar["source_sha256"]
        git_binding(sidecar["code_commit"], hashes)
        engine = {p: h for p, h in hashes.items() if p.startswith("src/graphmind/")}
        require_equal(
            "reviewed engine",
            engine,
            original_sources(CHECKOUT, sidecar["engine_review_commit"], engine),
        )
    else:
        binding = provenance["source_binding"]
        hashes = binding["source_sha256"]
        git_binding(binding["commit"], hashes)
    require_equal(
        "recorded measurement digests",
        {
            Path(p).name: h
            for p, h in hashes.items()
            if Path(p).parent.as_posix() == "benchmarks" and p.endswith(".py")
        },
        provenance["measurement_code"],
    )
    answer = read(root / "answers/binding.json")
    require_equal("reader binding timing", "pre-model-load", answer["assembly"])
    require_equal("reader retrieval binding", manifest_hash, answer["retrieval_manifest_sha256"])
    require_equal("reader model", FLAN, answer["model"])
    require_equal("reader settings", SETTINGS, answer["settings"])
    for key in (
        "max_new_tokens",
        "num_beams",
        "do_sample",
        "pad_token_id",
        "eos_token_id",
        "decoder_start_token_id",
    ):
        require_equal("generation config " + key, SETTINGS[key], answer["generation_config"][key])
    code = answer["source_binding"]
    git_binding(code["commit"], code["source_sha256"])
    audit = read(root / "answers/weight-audit.json")
    require_equal(
        "weight audit timing",
        "post-run-independent-reload" if recorded else "before-generation",
        audit["assembly"],
    )
    require_equal(
        "weight source", FLAN["files"]["model.safetensors"]["sha256"], audit["checkpoint_sha256"]
    )
    require_equal(
        "load report",
        {k: [] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")},
        audit["loader_report"],
    )
    require_equal(
        "all loaded tensors",
        (190, 192),
        (audit["checkpoint_tensors"], audit["verified_state_tensors"]),
    )
    if recorded:
        # Measured predictions, timing and loader audit cannot be regenerated from source text.
        anchor = read(HERE / "vendor/recorded.json")
        check_inventory(root / "answers", anchor["answers"])


def vector_table(path, cases):
    expected = {"q:" + c["qid"] for c in cases} | {
        "d:" + d["id"] for c in cases for d in c["documents"]
    }
    found = {}
    counts = {"queries": 0, "documents": 0, "truncated_queries": 0, "truncated_documents": 0}
    names = set()
    for n in range(math.ceil(len(expected) / 512)):
        stem = f"{n:04}"
        names.update([stem + ".json.gz", stem + ".f32.gz"])
        keys = read(path / (stem + ".json.gz"))
        require_equal(stem + " vector shard size", min(512, len(expected) - n * 512), len(keys))
        data = np.frombuffer(
            gzip.decompress((path / (stem + ".f32.gz")).read_bytes()), dtype="<f4"
        ).reshape(len(keys), WIDTH)
        if not np.allclose(np.linalg.norm(data, axis=1), 1, rtol=0, atol=2e-6):
            raise ValueError("invalid unit embedding")
        for (key, length), vec in zip(keys, data, strict=True):
            if key in found or key not in expected or type(length) is not int or length < 1:
                raise ValueError("invalid embedding key/length")
            found[key] = vec
            group = "queries" if key.startswith("q:") else "documents"
            counts[group] += 1
            counts["truncated_" + group] += int(length > 256)
    require_equal("vector file inventory", names, {p.name for p in path.iterdir()})
    require_equal("vector membership", expected, found.keys())
    require_equal("truncation counts", counts, read(path.parent / "truncation.json"))
    return found


def keyed(rows):
    by_id = {r["qid"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate question row")
    return by_id


def fixed_shards(path, rows, size=64):
    stems = {f"{i // size:04}.jsonl.gz" for i in range(0, len(rows), size)}
    require_equal(str(path) + " shard inventory", stems, {p.name for p in path.iterdir()})
    for i in range(0, len(rows), size):
        with gzip.open(path / f"{i // size:04}.jsonl.gz", "rt", encoding="utf8") as f:
            actual = [json.loads(line) for line in f]
        require_equal(
            f"{path.name}/{i // size:04} reconstructed records", rows[i : i + size], actual
        )


def timing(path, qids):
    rows = list(table(path / "timings"))
    fixed_shards(path / "timings", rows)
    ids = selection(qids, "graphmind-v1-timing:", 64)
    require_equal("timed queries", ids, [r["qid"] for r in rows])

    def percentiles(vals):
        if any(type(n) is not int or n <= 0 for n in vals):
            raise ValueError("invalid timing sample")
        return {
            "n": len(vals),
            "p50_ns": float(np.median(vals)),
            "p95_ns": float(np.quantile(vals, 0.95)),
        }

    for r in rows:
        require_equal("timing methods", set(METHODS), set(r["warmup_ns"]))
        percentiles(list(r["warmup_ns"].values()))
        require_equal(
            "timing repetition counts",
            {m: 5 for m in METHODS},
            {m: len(v) for m, v in r["samples_ns"].items()},
        )
    summary = {
        "query": {m: percentiles([t for r in rows for t in r["samples_ns"][m]]) for m in METHODS},
        "setup": {
            k: percentiles([r["setup_ns"][k] for r in rows]) for k in ("graph", "dense_index")
        },
    }
    require_equal("timing percentiles", summary, read(path / "timing-summary.json"))


def replay_split(base, ds, split, ids, frozen, tmp):
    path = base / ds / split
    cases = canonical_rows(base, ds, split, "cases")
    labs = canonical_rows(base, ds, split, "judgments")
    for name, rows in (("cases", cases), ("judgments", labs)):
        fixed_shards(path / name, rows)
        require_equal(name + " membership", ids, [r["qid"] for r in rows])
    labels = keyed(labs)
    matched = [(c, labels[c["qid"]]) for c in cases]
    vectors = vector_table(path / "vectors", cases)
    runs = keyed(list(table(path / "runs")))
    require_equal("run membership", set(ids), runs.keys())

    def prepare():
        for c in cases:
            q = c["qid"]
            qvec = vectors["q:" + q].astype(np.float64)
            docs = c["documents"]
            dense = runs[q]["scores"]["dense"]
            ids_and_scores = {d["id"]: v for d, v in zip(docs, dense, strict=True)}
            for pid, score in ids_and_scores.items():
                independent = float(qvec @ vectors["d:" + pid].astype(np.float64))
                if not math.isfinite(score) or abs(independent - score) > 3e-7:
                    raise ValueError(f"{q} dense score differs from vectors")
            yield build_graph(docs), bm25(docs, c["question"]), ids_and_scores

    prepared = prepare()
    if split == "train":
        prepared = list(prepared)
        chosen = tune(matched, prepared, tmp)
        require_equal("training-selected config", frozen, {k: list(v) for k, v in chosen.items()})
        fixed_shards(path / "grid", list(table(tmp / "grid")))
        require_equal(
            "grid summary", read(tmp / "grid-summary.json"), read(path / "grid-summary.json")
        )
    pairs = run_split(matched, prepared, tmp, frozen)
    for name in ("runs", "metrics"):
        fixed_shards(path / name, list(table(tmp / name)))
    for name in ("aggregate.json", "graph-counts.json", "crosscheck.json"):
        require_equal(name, read(tmp / name), read(path / name))
    directories = {
        "cases",
        "judgments",
        "vectors",
        "runs",
        "metrics",
        "truncation.json",
        "aggregate.json",
        "graph-counts.json",
        "crosscheck.json",
    }
    if split == "train":
        directories.update(["grid", "grid-summary.json"])
    else:
        directories.update(["timings", "timing-summary.json"])
        timing(path, ids)
    require_equal(ds + "/" + split + " contents", directories, {p.name for p in path.iterdir()})
    if split == "dev":
        return summarize(pairs, components([lab["support_ids"] for _, lab in matched]))
    return None


def replay_retrieval(base):
    frozen = read(base / "frozen.json")
    require_equal(
        "retrieval datasets",
        set(COUNTS)
        | {"primary.json", "manifest.json", "frozen.json", "elapsed.json", "provenance.json"},
        {p.name for p in base.iterdir()},
    )
    primary = {}
    for ds, counts in COUNTS.items():
        info = read(base / ds / "membership.json.gz")
        audit = read(base / ds / "source-audit.json")
        for label, value in (("membership", info), ("source_audit", audit)):
            require_equal(
                ds + " " + label,
                CANONICAL[ds][label],
                hashlib.sha256(json_bytes(value)).hexdigest(),
            )
        tune_ids, dev_ids = membership(info["train_ids"], info["dev_ids"])
        require_equal("split sizes", counts, (len(info["train_ids"]), len(dev_ids)))
        require_equal("selected training", tune_ids, info["tune_ids"])
        require_equal(
            ds + " retrieval inventory",
            {"membership.json.gz", "source-audit.json", "train", "dev"},
            {p.name for p in (base / ds).iterdir()},
        )
        for split, ids in (("train", tune_ids), ("dev", dev_ids)):
            with tempfile.TemporaryDirectory() as tmp:
                print(f"Replaying {ds}/{split}, {len(ids)} questions", flush=True)
                result = replay_split(base, ds, split, ids, frozen[ds], Path(tmp))
                if result is not None:
                    primary[ds] = result
    for result, adjusted in zip(
        primary.values(), holm([r["p"] for r in primary.values()]), strict=True
    ):
        result["p_holm"] = adjusted
    require_equal("retrieval paired estimates", primary, read(base / "primary.json"))


def check_prediction(tok, row):
    expected = check_output(tok, row["decoder_ids"])
    require_equal("decoded reader output", expected, {k: row[k] for k in expected})
    require_equal("prediction fields", set(expected) | {"qid", "method", "decoder_ids"}, set(row))


def replay_answers(root):
    tok = tokenizer()
    path = root / "answers"
    ids, inputs = answer_inputs(root / "retrieval", tok)
    require_equal("answer selection", ids, read(path / "selection.json"))
    binding = read(path / "binding.json")
    predictions = {}
    for ds, rows in inputs.items():
        require_equal(ds + " bound prompts", binding["inputs_sha256"][ds], bound_rows(rows))
        fixed_shards(path / ds / "inputs", rows)
        preds = list(table(path / ds / "predictions"))
        fixed_shards(path / ds / "predictions", preds)
        require_equal(
            ds + " prediction joins",
            [(r["qid"], r["method"]) for r in rows],
            [(r["qid"], r["method"]) for r in preds],
        )
        for r in preds:
            check_prediction(tok, r)
        predictions[ds] = preds
        require_equal(
            ds + " answer inventory",
            {"inputs", "predictions", "metrics"},
            {p.name for p in (path / ds).iterdir()},
        )
    pairs, summary = analyze(predictions, judgments(root / "retrieval"), ids)
    for ds, rows in pairs.items():
        fixed_shards(path / ds / "metrics", rows)
    require_equal("reader metrics and inference", summary, read(path / "summary.json"))
    require_equal(
        "answer directory inventory",
        set(COUNTS)
        | {"selection.json", "binding.json", "elapsed.json", "weight-audit.json", "summary.json"},
        {p.name for p in path.iterdir()},
    )
    index, graphs = explorer(root)
    require_equal("explorer index", index, read(root / "explorer-index.json"))
    require_equal("explorer datasets", set(COUNTS), {p.name for p in (root / "explorer").iterdir()})
    for ds, graphs_for_ds in graphs.items():
        fixed_shards(root / "explorer" / ds, graphs_for_ds, size=8)


def verify(root, recorded=True):
    expected = {"release.json", "retrieval", "answers", "explorer", "explorer-index.json"}
    if recorded:
        require_equal("original release manifest", ORIGINAL_RELEASE, digest(root / "release.json"))
        expected.add("retrieval-provenance.json")
    require_equal("release root inventory", expected, {p.name for p in root.iterdir()})
    manifest = read(root / "release.json")
    require_equal("release manifest fields", {"schema", "files"}, set(manifest))
    require_equal("release schema", 1, manifest["schema"])
    check_inventory(root, manifest["files"], ("release.json",))
    checked_sources(root, recorded)
    replay_answers(root)
    replay_retrieval(root / "retrieval")
    print(
        "Verified 10,822 retrieval cases, 9,822 development memberships, 200 explorer graphs and 600 matched predictions.",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results"))
    p.add_argument(
        "--reproduction",
        action="store_true",
        help="verify a separately measured run with pre-run source binding",
    )
    args = p.parse_args()
    verify(args.root, not args.reproduction)


if __name__ == "__main__":
    main()
