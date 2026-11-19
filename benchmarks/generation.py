"""Matched prompt construction, output decoding and question-paired answer analysis."""

import json
import math
import re
import string
from collections import Counter

from .sources import HERE, verified
from .statistics import components, holm

VENDOR = HERE / "vendor"
FLAN = json.loads((VENDOR / "flan/files.json").read_text())
ANSWER_METHODS = ("dense", "hybrid", "graph-2")
SEED = 20261119
SETTINGS = {
    "input_limit": 512,
    "passage_limit": 96,
    "context_limit": 384,
    "top_k": 4,
    "max_new_tokens": 48,
    "num_beams": 1,
    "do_sample": False,
    "pad_token_id": 0,
    "decoder_start_token_id": 0,
    "eos_token_id": 1,
    "batch": 1,
    "dtype": "float32",
    "device": "cpu",
    "attention": "eager",
    "threads": 6,
    "interop_threads": 1,
    "deterministic_algorithms": True,
}


def tokenizer():
    from tokenizers import Tokenizer

    path = verified(VENDOR / "flan/tokenizer.json", FLAN["files"]["tokenizer.json"])
    return Tokenizer.from_file(str(path))


def prefix(tok, text, limit):
    """Trim only at source code-point boundaries and re-tokenize each chosen prefix."""
    encoding = tok.encode(text, add_special_tokens=False)
    if len(encoding.ids) <= limit:
        return text
    stop = encoding.offsets[limit - 1][1]
    while stop and len(tok.encode(text[:stop], add_special_tokens=False).ids) > limit:
        stop -= 1
    return text[:stop]


def build_prompt(tok, question, documents):
    if not 1 <= len(documents) <= 4 or len({d["id"] for d in documents}) != len(documents):
        raise ValueError("expected up to four distinct ranked documents")
    head = prefix(tok, "Answer the question briefly. Question: " + question, 118) + "\nContext:\n"
    context = ""
    passages = []
    for doc in documents:
        label = "Title: "
        full = label + doc["title"] + "\nPassage: " + doc["text"]
        part = prefix(tok, full, SETTINGS["passage_limit"])
        start = len(head) + len(context)
        body_start = min(len(part), len(full) - len(doc["text"]))
        passages.append(
            {
                "id": doc["id"],
                "title_end": min(len(doc["title"]), max(0, len(part) - len(label))),
                "body_end": len(part) - body_start,
                "prompt_span": [start + body_start, start + len(part)],
                "tokens": len(tok.encode(part, add_special_tokens=False).ids),
            }
        )
        context += part + "\n"
    context = context.removesuffix("\n")
    text = head + context + "\nAnswer:"
    ids = tok.encode(text).ids
    context_tokens = len(tok.encode(context, add_special_tokens=False).ids)
    if context_tokens > SETTINGS["context_limit"] or len(ids) > SETTINGS["input_limit"]:
        raise ValueError("final retokenization exceeds protocol budget")
    return {"text": text, "input_ids": ids, "context_tokens": context_tokens, "passages": passages}


def check_output(tok, ids):
    """Sequences contain the decoder start and unpadded tokens for one example."""
    if (
        not 2 <= len(ids) <= 49
        or ids[0] != 0
        or any(type(i) is not int or not 0 <= i < tok.get_vocab_size() for i in ids)
        or 1 in ids[1:-1]
        or (ids[-1] != 1 and len(ids) != 49)
    ):
        raise ValueError("invalid decoder IDs or stopping point")
    return {
        "answer": tok.decode(ids[1:], skip_special_tokens=True),
        "stop": "eos" if ids[-1] == 1 else "length",
    }


def check_predictions(keys, expected):
    if Counter(keys) != Counter(expected) or any(n != 1 for n in Counter(keys).values()):
        raise ValueError("prediction membership incomplete or duplicated")


def score(dataset, prediction, references):
    def tokens(text):
        no_punctuation = text.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\b(a|an|the)\b", " ", no_punctuation).split()

    if not references or dataset not in ("hotpot", "musique"):
        raise ValueError("missing references or unknown dataset")
    pred = tokens(prediction)
    em = 0
    f1 = 0.0
    special = (["yes"], ["no"], ["noanswer"])
    for ref in references:
        gold = tokens(ref)
        em = max(em, int(pred == gold))
        if dataset == "hotpot" and (gold in special or pred in special) and pred != gold:
            continue
        same = sum((Counter(pred) & Counter(gold)).values())
        if same:
            f1 = max(f1, 2 * same / (len(pred) + len(gold)))
        elif dataset == "musique" and not pred and not gold:
            f1 = 1.0
    return {"em": em, "f1": f1}


def official_score(dataset, prediction, references):
    if dataset == "hotpot":
        from .vendor.hotpot.hotpot_evaluate_v1 import exact_match_score, f1_score

        (gold,) = references
        return {
            "em": float(exact_match_score(prediction, gold)),
            "f1": f1_score(prediction, gold)[0],
        }
    from .vendor.musique.metrics.answer import AnswerMetric

    evaluator = AnswerMetric()
    evaluator(prediction, references)
    em, f1 = evaluator.get_metric()
    return {"em": em, "f1": f1}


def paired(deltas, groups):
    import numpy as np

    x = np.asarray(deltas, dtype=np.float64)

    def interval(totals, sizes):
        rng = np.random.default_rng(SEED)
        n = len(sizes)
        indices = rng.integers(n, size=(10000, n))
        samples = np.asarray(totals)[indices].sum(axis=1) / np.asarray(sizes)[indices].sum(axis=1)
        return np.quantile(samples, [0.025, 0.975], method="linear").tolist()

    sizes = [len(g) for g in groups]
    rng = np.random.default_rng(SEED)
    signed = (rng.integers(2, size=(10000, len(x))) * 2 - 1) * x
    observed = abs(x.sum())
    extreme = (abs(signed.sum(axis=1)) >= observed - 1e-12).sum()
    return {
        "n": len(x),
        "wins": int((x > 0).sum()),
        "losses": int((x < 0).sum()),
        "ties": int((x == 0).sum()),
        "delta": math.fsum(deltas) / len(deltas),
        "p": (int(extreme) + 1) / 10001,
        "query_ci95": interval(x, [1] * len(x)),
        "cluster_ci95": interval([x[g].sum() for g in groups], sizes) if len(sizes) > 1 else None,
        "clusters": len(sizes),
        "cluster_sizes": sorted(sizes, reverse=True),
        "largest_cluster_share": max(sizes) / len(x),
    }


def analyze(predictions, judgments, ids):
    summaries = {}
    pairs = {}
    for ds, qids in ids.items():
        scores = {}
        max_error = {"em": 0.0, "f1": 0.0}
        rows = predictions[ds]
        check_predictions(
            [(r["qid"], r["method"]) for r in rows], [(q, m) for q in qids for m in ANSWER_METHODS]
        )
        for r in rows:
            qid, method = r["qid"], r["method"]
            refs = judgments[ds][qid]["answers"]
            values = score(ds, r["answer"], refs)
            independent = official_score(ds, r["answer"], refs)
            for k, v in independent.items():
                max_error[k] = max(max_error[k], abs(v - values[k]))
            scores[qid, method] = values
        if max(max_error.values()) > 1e-12:
            raise ValueError("official answer score disagrees")
        pairs[ds] = []
        for q in qids:
            methods = {m: scores[q, m] for m in ANSWER_METHODS}
            deltas = {
                a + "_minus_" + b: {k: methods[a][k] - methods[b][k] for k in ("em", "f1")}
                for a, b in (("graph-2", "hybrid"), ("graph-2", "dense"), ("hybrid", "dense"))
            }
            pairs[ds].append({"qid": q, "methods": methods, "deltas": deltas})
        all_qids = sorted(judgments[ds])
        groups = components([judgments[ds][q]["support_ids"] for q in all_qids])
        positions = {q: i for i, q in enumerate(qids)}
        occupied = [[positions[all_qids[i]] for i in g if all_qids[i] in positions] for g in groups]
        primary = paired(
            [r["deltas"]["graph-2_minus_hybrid"]["f1"] for r in pairs[ds]],
            [g for g in occupied if g],
        )
        summaries[ds] = {
            "n": len(qids),
            "methods": {
                m: {k: math.fsum(scores[q, m][k] for q in qids) / len(qids) for k in ("em", "f1")}
                for m in ANSWER_METHODS
            },
            "primary_graph2_minus_hybrid_f1": primary,
            "exploratory_mean_deltas": {
                c: {
                    k: math.fsum(r["deltas"][c][k] for r in pairs[ds]) / len(qids)
                    for k in pairs[ds][0]["deltas"][c]
                    if (c, k) != ("graph-2_minus_hybrid", "f1")
                }
                for c in pairs[ds][0]["deltas"]
            },
            "crosscheck": {
                "predictions": len(rows),
                "scalar_comparisons": 2 * len(rows),
                "max_abs": max_error,
            },
        }
    primary = [s["primary_graph2_minus_hybrid_f1"] for s in summaries.values()]
    for p, adj in zip(primary, holm([p["p"] for p in primary]), strict=True):
        p["p_holm"] = adj
    return pairs, summaries
