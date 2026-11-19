"""Translate source-specific provenance and judgments into disjoint case records."""

import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from graphmind import Document, document_id
from graphmind.documents import validate_documents, validate_question

from .sources import DATA

COUNTS = {"hotpot": (90447, 7405), "musique": (19938, 2417)}


def selection(ids, prefix, n):
    return sorted(ids, key=lambda i: (hashlib.sha256((prefix + i).encode()).digest(), i))[:n]


def membership(train, dev):
    if len(set(train)) != len(train) or len(set(dev)) != len(dev) or set(train) & set(dev):
        raise ValueError("question IDs repeat within or across splits")
    if len(train) < 500:
        raise ValueError("need at least 500 training IDs")
    return selection(train, "graphmind-v1-tune:", 500), sorted(dev)


def raw_records(cache, dataset, split):
    if dataset == "hotpot":
        import pyarrow.parquet as pq

        label = "train" if split == "train" else "validation"
        for spec in DATA:
            path = Path(spec["path"])
            if path.name.startswith("hotpot-" + label):
                for batch in pq.ParquetFile(cache / path.name).iter_batches(batch_size=256):
                    yield from batch.to_pylist()
    elif dataset == "musique":
        with (cache / f"musique_ans_v1.0_{split}.jsonl").open(encoding="utf8") as f:
            for line in f:
                yield json.loads(line)
    else:
        raise ValueError("unknown dataset")


def hotpot_parts(row):
    ctx = row["context"]
    entries = list(zip(ctx["title"], ctx["sentences"], strict=True))
    facts = row["supporting_facts"]
    support_titles = set(facts["title"])
    if not support_titles.issubset(t for t, _ in entries):
        raise ValueError("missing supporting title")
    facts = list(zip(facts["title"], facts["sent_id"], strict=True))
    bad = [
        [t, s]
        for t, s in facts
        if type(s) != int or not any(t == title and 0 <= s < len(group) for title, group in entries)
    ]
    paragraphs = [
        (i, title, "".join(sentences), title in support_titles)
        for i, (title, sentences) in enumerate(entries)
    ]
    return paragraphs, row["type"], [row["answer"]], bad


def musique_parts(row):
    if row["answerable"] is not True:
        raise ValueError("MuSiQue-Ans requires an answerable question")
    paragraphs = [
        (p["idx"], p["title"], p["paragraph_text"], p["is_supporting"]) for p in row["paragraphs"]
    ]
    indices = [i for i, _, _, _ in paragraphs]
    support = {i for i, _, _, s in paragraphs if s}
    decomposition = [s["paragraph_support_idx"] for s in row["question_decomposition"]]
    if (
        any(type(i) != int or i < 0 for i in indices + decomposition)
        or len(indices) != len(set(indices))
        or set(decomposition) != support
        or any(type(s) != bool for _, _, _, s in paragraphs)
    ):
        raise ValueError("invalid or inconsistent paragraph support index")
    return (
        paragraphs,
        row["id"].split("hop", 1)[0] + "hop",
        [row["answer"], *row["answer_aliases"]],
        [],
    )


def adapt(dataset, row):
    paragraphs, stratum, answers, bad = {"hotpot": hotpot_parts, "musique": musique_parts}[dataset](
        row
    )
    docs = {}
    mapping = []
    support = []
    for i, title, text, positive in paragraphs:
        pid = document_id(title, text)
        docs[pid] = Document(pid, title, text)
        mapping.append([i, pid])
        if positive:
            support.append(pid)
    if not support:
        raise ValueError("record has no support")
    if not isinstance(row["id"], str) or not row["id"].strip():
        raise ValueError("invalid question ID")
    case = {
        "qid": row["id"],
        "question": validate_question(row["question"]),
        "documents": [asdict(d) for d in validate_documents(list(docs.values()))],
        "source_map": sorted(mapping),
    }
    labels = {
        "qid": row["id"],
        "support_ids": sorted(set(support)),
        "source_support_count": len(support),
        "stratum": stratum,
        "answers": answers,
        "invalid_sentence_facts": bad,
    }
    return case, labels


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def inventory(cache, dataset):
    """Scan all source identities, storing public text only for the experiment membership."""
    train_ids = [r["id"] for r in raw_records(cache, dataset, "train")]
    dev_ids = [r["id"] for r in raw_records(cache, dataset, "dev")]
    if (len(train_ids), len(dev_ids)) != COUNTS[dataset]:
        raise ValueError("unexpected source split size")
    tune, dev = membership(train_ids, dev_ids)
    selected = set(tune + dev)
    splits = {}
    norms = {}
    pids = {}
    stats = {}
    for split in ["train", "dev"]:
        rows = {}
        questions = Counter()
        identities = set()
        anomalies = []
        for raw in raw_records(cache, dataset, split):
            parts, _, _, bad = (hotpot_parts if dataset == "hotpot" else musique_parts)(raw)
            questions[normalized(raw["question"])] += 1
            identities.update(document_id(t, b) for _, t, b, _ in parts)
            if bad:
                anomalies.append({"qid": raw["id"], "facts": bad})
            if raw["id"] in selected:
                rows[raw["id"]] = adapt(dataset, raw)
        splits[split] = [rows[i] for i in (tune if split == "train" else dev)]
        norms[split] = questions
        pids[split] = identities
        stats[split] = {
            "rows": sum(questions.values()),
            "distinct_paragraphs": len(identities),
            "repeated_normalized_question_excess": sum(n - 1 for n in questions.values()),
            "invalid_sentence_questions": len(anomalies),
            "invalid_sentence_indices": sum(len(a["facts"]) for a in anomalies),
            "sentence_anomalies": anomalies,
        }

    def overlap(train_questions, train_pids):
        return {
            "normalized_questions": len(set(train_questions) & set(norms["dev"])),
            "paragraph_ids": len(train_pids & pids["dev"]),
        }

    stats["train_dev_overlap"] = overlap(norms["train"], pids["train"])
    stats["tune_dev_overlap"] = overlap(
        [normalized(c["question"]) for c, _ in splits["train"]],
        {d["id"] for c, _ in splits["train"] for d in c["documents"]},
    )
    return splits, {"train_ids": sorted(train_ids), "dev_ids": dev, "tune_ids": tune}, stats
