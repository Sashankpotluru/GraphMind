"""Sparse lexical scoring and finite, explicitly attributable paragraph walks."""

import math
import re
from collections import Counter
from collections.abc import Mapping
from typing import TypedDict

from .documents import STOPWORDS, Documents, GraphMindError, validate_documents, validate_question
from .graph import Graph, build_graph

type Scores = dict[str, float]


class Ranked(TypedDict):
    id: str
    score: float
    base_score: float
    graph_score: float


class Expansion(TypedDict):
    ranking: list[Ranked]
    hops: list[Scores]
    per_seed: dict[str, Scores]
    seed_weights: Scores
    depth: int
    mix: float
    seeds: int
    effective_mix: float


class Retrieval(Expansion):
    graph: Graph


def _tokens(text: str) -> list[str]:
    folded = (word.casefold() for word in re.findall(r"\w+", text))
    return [token for token in folded if token not in STOPWORDS]


def bm25(documents: Documents, question: str) -> Scores:
    docs = validate_documents(documents)
    terms = sorted(set(_tokens(validate_question(question))))
    counts = {d.id: Counter(_tokens(d.title + "\n" + d.text)) for d in docs}
    lengths = {pid: c.total() for pid, c in counts.items()}
    avg_len = sum(lengths.values()) / len(docs)
    scores = dict.fromkeys(counts, 0.0)
    for term in terms:
        df = sum(term in counter for counter in counts.values())
        if not df:
            continue
        idf = math.log1p((len(docs) - df + 0.5) / (df + 0.5))
        for pid, counter in counts.items():
            tf = counter[term]
            if tf:
                norm = 1.2 * (0.25 + 0.75 * lengths[pid] / avg_len)
                scores[pid] += idf * (tf * 2.2) / (tf + norm)
    return scores


def _integer(value: int, field: str, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise GraphMindError(field, f"expected an integer from {low} to {high}")


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraphMindError(field, "expected a finite nonnegative number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise GraphMindError(field, "number exceeds the finite float range") from exc
    if not math.isfinite(number) or number < 0:
        raise GraphMindError(field, "expected a finite nonnegative number")
    return number


def _normalize(scores: Scores) -> Scores:
    largest = max(scores.values(), default=0.0)
    if not largest:
        return scores.copy()
    # Scaling keeps sums finite even when individual scores approach float max.
    scaled = {pid: score / largest for pid, score in scores.items()}
    total = math.fsum(scaled.values())
    return {pid: value / total for pid, value in scaled.items()}


def _step(mass: Scores, neighbors: dict[str, list[str]]) -> Scores:
    result = dict.fromkeys(mass, 0.0)
    for pid, value in mass.items():
        targets = neighbors[pid]
        if targets:
            share = value / len(targets)
            for target in targets:
                result[target] += share
        else:
            result[pid] += value
    return result


def expand(
    graph: Graph,
    base_scores: Mapping[str, float],
    depth: int = 2,
    mix: float = 0.35,
    seeds: int = 2,
) -> Expansion:
    _integer(depth, "depth", 0, 3)
    _integer(seeds, "seeds", 1, 3)
    mix = _number(mix, "mix")
    if mix > 1:
        raise GraphMindError("mix", "expected a value between zero and one")
    ids = sorted(p["id"] for p in graph["paragraphs"])
    if not isinstance(base_scores, Mapping):
        raise GraphMindError("base_scores", "expected a mapping of paragraph IDs to numbers")
    for pid in base_scores:
        if pid not in ids:
            raise GraphMindError("base_scores", f"unknown paragraph ID {pid!r}")
    raw = {pid: _number(base_scores.get(pid, 0.0), "base_scores") for pid in ids}
    base = _normalize(raw)
    positive = sorted((pid for pid in ids if raw[pid] > 0), key=lambda pid: (-raw[pid], pid))
    weights = _normalize({pid: raw[pid] for pid in positive[:seeds]})
    neighbors: dict[str, list[str]] = {pid: [] for pid in ids}
    for edge in graph["projection"]:
        neighbors[edge["source"]].append(edge["target"])
        neighbors[edge["target"]].append(edge["source"])

    hop_weights = [0.5**t for t in range(depth + 1)]
    total = sum(hop_weights)
    hops: list[Scores] = [dict.fromkeys(ids, 0.0) for _ in hop_weights]
    per_seed: dict[str, Scores] = {}
    for seed, weight in weights.items():
        mass = dict.fromkeys(ids, 0.0)
        mass[seed] = weight
        contribution = dict.fromkeys(ids, 0.0)
        for t, hop_weight in enumerate(hop_weights):
            if t:
                mass = _step(mass, neighbors)
            for pid in ids:
                hops[t][pid] += mass[pid]
                contribution[pid] += mass[pid] * hop_weight / total
        per_seed[seed] = contribution

    effective = mix if depth and graph["projection"] and weights else 0.0
    ranking: list[Ranked] = []
    for pid in ids:
        walk = math.fsum(component[pid] for component in per_seed.values())
        score = (1 - effective) * base[pid] + effective * walk if effective else base[pid]
        ranking.append({"id": pid, "score": score, "base_score": base[pid], "graph_score": walk})
    ranking.sort(key=lambda row: (-row["score"], row["id"]))
    return {
        "ranking": ranking,
        "hops": hops,
        "per_seed": per_seed,
        "seed_weights": weights,
        "depth": depth,
        "mix": mix,
        "seeds": seeds,
        "effective_mix": effective,
    }


def retrieve(
    documents: Documents, question: str, depth: int = 2, mix: float = 0.35, seeds: int = 2
) -> Retrieval:
    scores = bm25(documents, question)
    graph = build_graph(documents)
    return {**expand(graph, scores, depth, mix, seeds), "graph": graph}
