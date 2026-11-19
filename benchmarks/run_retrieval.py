"""Reproduce the predeclared supplied-candidate retrieval benchmark."""

import argparse
import gzip
import itertools
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import median

from graphmind import bm25, build_graph, expand

from .data import COUNTS, inventory, selection
from .evaluate import KEYS, crosscheck, metrics
from .models import LIMIT, cosine_scores, load_encoder, model_identity
from .sources import DATA, MODEL, digest, verified
from .statistics import components, holm, summarize

HYBRID_GRID = list(itertools.product((0.25, 0.5, 0.75), (10, 60)))
GRAPH_GRID = list(itertools.product((0.15, 0.35, 0.60), (1, 2, 3)))
METHODS = ("bm25", "dense", "hybrid", "graph-1", "graph-2", "graph-3")
SHARD = 64


def ranked(scores):
    return sorted(scores, key=lambda p: (-scores[p], p))


def rrf(sparse, dense, weight, constant):
    rs = {pid: i for i, pid in enumerate(ranked(sparse), 1)}
    rd = {pid: i for i, pid in enumerate(ranked(dense), 1)}
    if rs.keys() != rd.keys():
        raise ValueError("fusion requires equal candidate sets")
    return {p: (1 - weight) / (constant + rs[p]) + weight / (constant + rd[p]) for p in rs}


def choose(objectives):
    return max(range(len(objectives)), key=objectives.__getitem__)


def json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def put_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def compressed(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("xb") as f,
        gzip.GzipFile(filename="", fileobj=f, mode="wb", compresslevel=6, mtime=0) as z,
    ):
        z.write(data)


def write_shards(path, rows):
    for i in range(0, len(rows), SHARD):
        compressed(
            path / f"{i // SHARD:04}.jsonl.gz",
            b"".join(json_bytes(r) + b"\n" for r in rows[i : i + SHARD]),
        )


def embeddings(cases, encoder, path):
    texts = {}
    for c in cases:
        texts["q:" + c["qid"]] = c["question"]
        for d in c["documents"]:
            texts["d:" + d["id"]] = d["title"] + "\n" + d["text"]
    # Similar lengths limit padding. The textual key breaks length ties.
    keys = sorted(texts, key=lambda k: (len(texts[k]), k))
    out = {}
    trunc = Counter()
    for i in range(0, len(keys), 512):
        chunk = keys[i : i + 512]
        matrix, lengths = encoder.encode([texts[k] for k in chunk])
        stem = f"{i // 512:04}"
        compressed(path / (stem + ".f32.gz"), matrix.astype("<f4", copy=False).tobytes("C"))
        compressed(path / (stem + ".json.gz"), json_bytes(list(zip(chunk, lengths, strict=True))))
        for key, vec, n in zip(chunk, matrix, lengths, strict=True):
            out[key] = vec
            trunc["queries" if key.startswith("q:") else "documents"] += 1
            if n > LIMIT:
                trunc["truncated_queries" if key.startswith("q:") else "truncated_documents"] += 1
        if i == 0 or i // 512 % 8 == 7 or i + 512 >= len(keys):
            print(
                f"{path.parent.parent.name}/{path.parent.name} embeddings {min(i + 512, len(keys))}/{len(keys)}",
                flush=True,
            )
    return out, {
        k: trunc[k] for k in ("queries", "documents", "truncated_queries", "truncated_documents")
    }


def dense_for(case, vectors):
    import numpy as np

    ids = [d["id"] for d in case["documents"]]
    docs = np.stack([vectors["d:" + p] for p in ids])
    return cosine_scores(ids, docs, vectors["q:" + case["qid"]])


def prep(case, vectors):
    return (
        build_graph(case["documents"]),
        bm25(case["documents"], case["question"]),
        dense_for(case, vectors),
    )


def mini_graph(expansion, ids):
    # Coordinates are shared across methods. Never round the scores.
    ranking = {r["id"]: r for r in expansion["ranking"]}
    return {
        "effective_mix": expansion["effective_mix"],
        **{key: [ranking[p][key] for p in ids] for key in ("score", "base_score", "graph_score")},
        "hops": [[h[p] for p in ids] for h in expansion["hops"]],
        "seed_weights": expansion["seed_weights"],
        "per_seed": {p: [s[i] for i in ids] for p, s in expansion["per_seed"].items()},
    }


def evaluate_case(case, prepared, config):
    graph, sparse, dense = prepared
    ids = [d["id"] for d in case["documents"]]
    positions = {p: i for i, p in enumerate(ids)}
    fusion = rrf(sparse, dense, *config["hybrid"])
    scores = {"bm25": sparse, "dense": dense, "hybrid": fusion}
    ranks = {m: ranked(s) for m, s in scores.items()}
    graphs = {}
    for depth in (1, 2, 3):
        method = f"graph-{depth}"
        ex = expand(graph, fusion, depth, *config["graph"])
        ranks[method] = [row["id"] for row in ex["ranking"]]
        graphs[method] = mini_graph(ex, ids)
    stats = {k: len(v) for k, v in graph.items()}
    stats["unique_mention_pairs"] = len(
        {(m["paragraph_id"], m["entity_id"]) for m in graph["mentions"]}
    )
    return {
        "qid": case["qid"],
        "scores": {m: [s[p] for p in ids] for m, s in scores.items()},
        "rankings": {m: [positions[p] for p in order] for m, order in ranks.items()},
        "graph": graphs,
        "graph_counts": stats,
    }, ranks


def tune(rows, prepared, path):
    outcomes = [{"qid": c["qid"]} for c, _ in rows]
    grid_report = {}
    config = {}
    for name, grid in (("hybrid", HYBRID_GRID), ("graph", GRAPH_GRID)):
        objectives = []
        grid_report[name] = []
        for settings in grid:
            cell = []
            for i, ((_, labels), (g, s, d)) in enumerate(zip(rows, prepared, strict=True)):
                if name == "hybrid":
                    order = ranked(rrf(s, d, *settings))
                else:
                    hybrid = rrf(s, d, *config["hybrid"])
                    order = [r["id"] for r in expand(g, hybrid, 2, *settings)["ranking"]]
                values = metrics(order, labels["support_ids"])
                outcomes[i].setdefault(name, []).append(values)
                cell.append(values)
            avg = {k: math.fsum(r[k] for r in cell) / len(cell) for k in KEYS}
            grid_report[name].append({"settings": settings, "mean": avg})
            objectives.append((avg["all@4"], avg["ndcg@4"]))
        config[name] = grid[choose(objectives)]
    write_shards(path / "grid", outcomes)
    put_json(path / "grid-summary.json", grid_report)
    return config


def run_split(rows, prepared, path, config):
    counts = Counter()
    worst = Counter()
    summaries = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    pairs = []
    graph_totals = Counter()
    runs = []
    metric_rows = []
    for i, ((case, labels), inputs) in enumerate(zip(rows, prepared, strict=True)):
        run, ranks = evaluate_case(case, inputs, config)
        graph_totals.update(run["graph_counts"])
        one = {"qid": case["qid"], "methods": {}}
        for method, order in ranks.items():
            values = metrics(order, labels["support_ids"])
            one["methods"][method] = values
            for key, error in crosscheck(order, labels["support_ids"], values).items():
                counts[key] += 1
                worst[key] = max(worst[key], error)
                if error > 1e-10:
                    raise ValueError(
                        f"independent metric disagreement {case['qid']} {method} {key} {error}"
                    )
            for stratum in ("overall", labels["stratum"]):
                for k, v in values.items():
                    summaries[stratum][method][k].append(v)
        delta = one["methods"]["graph-2"]["all@4"] - one["methods"]["hybrid"]["all@4"]
        pairs.append(delta)
        one["graph2_minus_hybrid_all4"] = delta
        runs.append(run)
        metric_rows.append(one)
        if len(runs) == SHARD or i + 1 == len(rows):
            for folder, batch in [("runs", runs), ("metrics", metric_rows)]:
                compressed(
                    path / folder / f"{i // SHARD:04}.jsonl.gz",
                    b"".join(json_bytes(r) + b"\n" for r in batch),
                )
            runs.clear()
            metric_rows.clear()
        if (i + 1) % 1024 == 0:
            print(f"{path.parent.name}/{path.name} scored {i + 1}/{len(rows)}", flush=True)
    aggregate = {
        s: {
            "n": len(next(iter(ms.values()))["all@4"]),
            "methods": {
                m: {k: math.fsum(v) / len(v) for k, v in ks.items()} for m, ks in ms.items()
            },
        }
        for s, ms in summaries.items()
    }
    put_json(path / "aggregate.json", aggregate)
    put_json(
        path / "crosscheck.json",
        {"count": dict(counts), "max_abs": dict(worst), "scalars": sum(counts.values())},
    )
    put_json(path / "graph-counts.json", dict(graph_totals))
    return pairs


def timed(call, *args):
    start = time.perf_counter_ns()
    result = call(*args)
    return result, time.perf_counter_ns() - start


def time_case(case, encoder, config):
    docs = case["documents"]
    pids = [d["id"] for d in docs]
    graph, graph_ns = timed(lambda: build_graph(docs))
    (matrix, _), index_ns = timed(
        lambda: encoder.encode([d["title"] + "\n" + d["text"] for d in docs])
    )

    def query(method):
        if method != "dense":
            sparse = bm25(docs, case["question"])
            if method == "bm25":
                return ranked(sparse)
        vectors, _ = encoder.encode([case["question"]])
        dense = cosine_scores(pids, matrix, vectors[0])
        if method == "dense":
            return ranked(dense)
        hybrid = rrf(sparse, dense, *config["hybrid"])
        if method == "hybrid":
            return ranked(hybrid)
        return [
            r["id"] for r in expand(graph, hybrid, int(method[-1]), *config["graph"])["ranking"]
        ]

    samples = {m: [] for m in METHODS}
    warmup = {}
    for repeat in range(6):
        for m in METHODS:
            _, duration = timed(query, m)
            if repeat == 0:
                warmup[m] = duration
            else:
                samples[m].append(duration)
    return {
        "qid": case["qid"],
        "setup_ns": {"graph": graph_ns, "dense_index": index_ns},
        "warmup_ns": warmup,
        "samples_ns": samples,
    }


def timing(rows, encoder, config, root):
    """Execute each method separately, including its query-side work."""
    import numpy as np

    by_id = {c["qid"]: c for c, _ in rows}
    ids = selection(by_id, "graphmind-v1-timing:", 64)
    output = [time_case(by_id[qid], encoder, config) for qid in ids]
    write_shards(root / "timings", output)

    def percentiles(vals):
        return {
            "n": len(vals),
            "p50_ns": float(median(vals)),
            "p95_ns": float(np.quantile(vals, 0.95, method="linear")),
        }

    put_json(
        root / "timing-summary.json",
        {
            "query": {
                m: percentiles([t for r in output for t in r["samples_ns"][m]]) for m in METHODS
            },
            "setup": {
                k: percentiles([r["setup_ns"][k] for r in output]) for k in ("graph", "dense_index")
            },
        },
    )


def platform_info():
    info = {
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "processor": platform.processor(),
        "packages": {},
    }
    if platform.system() == "Darwin":
        for k in ["machdep.cpu.brand_string", "hw.memsize", "hw.physicalcpu"]:
            info[k] = subprocess.check_output(["sysctl", "-n", k], text=True).strip()
    for package in (
        "numpy",
        "scipy",
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "pytrec_eval-terrier",
        "pyarrow",
    ):
        info["packages"][package] = version(package)
    return info


def measure(cache, root):
    start = time.monotonic()
    for spec in DATA + MODEL:
        verified(cache / Path(spec["path"]).relative_to("data/cache"), spec)
    settings, identity = model_identity()
    put_json(
        root / "provenance.json",
        {
            "schema": 1,
            "model": settings,
            "encoder_identity": identity,
            "datasets": DATA,
            "hardware": platform_info(),
            "measurement_code": {
                p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        },
    )
    splits = {}
    frozen = {}
    primary = {}
    for dataset in COUNTS:
        splits[dataset], identities, info = inventory(cache, dataset)
        base = root / dataset
        compressed(base / "membership.json.gz", json_bytes(identities))
        put_json(base / "source-audit.json", info)
        for split, rows in splits[dataset].items():
            write_shards(base / split / "cases", [c for c, _ in rows])
            write_shards(base / split / "judgments", [j for _, j in rows])
        print(f"verified membership and paragraph provenance: {dataset}", flush=True)
    with load_encoder(cache) as encoder:
        for split in ("train", "dev"):
            if split == "dev":
                # Dev inference and scoring read the frozen configuration from disk.
                put_json(root / "frozen.json", frozen)
                frozen = json.loads((root / "frozen.json").read_text())
                print(
                    "both dataset configurations frozen; starting official development", flush=True
                )
            for dataset in COUNTS:
                rows = splits[dataset][split]
                path = root / dataset / split
                vectors, trunc = embeddings([c for c, _ in rows], encoder, path / "vectors")
                put_json(path / "truncation.json", trunc)
                prepared = (prep(c, vectors) for c, _ in rows)
                if split == "train":
                    prepared = list(prepared)
                    frozen[dataset] = tune(rows, prepared, path)
                    print(f"{dataset} training selection: {frozen[dataset]}", flush=True)
                deltas = run_split(rows, prepared, path, frozen[dataset])
                del vectors, prepared
                if split == "dev":
                    groups = components([lab["support_ids"] for _, lab in rows])
                    primary[dataset] = summarize(deltas, groups)
                    print(f"running warm sequential timings: {dataset}", flush=True)
                    timing(rows, encoder, frozen[dataset], path)
        for ds, p in zip(COUNTS, holm([primary[d]["p"] for d in COUNTS]), strict=True):
            primary[ds]["p_holm"] = p
        put_json(root / "primary.json", primary)
    put_json(root / "elapsed.json", {"measurement_seconds": time.monotonic() - start})
    files = [
        {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    put_json(root / "manifest.json", {"schema": 1, "files": files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error("output exists; choose a new directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.parent / ("." + out.name + ".lock")
    with lock.open("x") as handle:
        try:
            stage = Path(tempfile.mkdtemp(prefix="." + out.name + "-incomplete-", dir=out.parent))
            handle.write(stage.name)
            handle.flush()
            print(f"staging observations in {stage.name}", flush=True)
            measure(args.cache, stage)
            if out.exists():
                raise FileExistsError(out)
            stage.rename(out)
        finally:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
