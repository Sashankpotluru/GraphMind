"""Binary paragraph-level metrics and a separately implemented trec_eval/set check."""

import math

KEYS = ("all@2", "recall@2", "all@4", "recall@4", "all@6", "recall@6", "ndcg@4", "mrr")


def metrics(ranking, support):
    relevant = set(support)
    if not relevant or len(set(ranking)) != len(ranking) or not relevant.issubset(ranking):
        raise ValueError("incomplete ranking or invalid relevance judgments")
    hits = [i for i, p in enumerate(ranking, 1) if p in relevant]
    result = {}
    for k in (2, 4, 6):
        count = sum(r <= k for r in hits)
        result[f"all@{k}"] = int(count == len(relevant))
        result[f"recall@{k}"] = count / len(relevant)
    ideal = sum(1 / math.log2(1 + i) for i in range(1, min(4, len(relevant)) + 1))
    result["ndcg@4"] = sum(1 / math.log2(1 + i) for i in hits if i <= 4) / ideal
    result["mrr"] = 1 / hits[0]
    return result


def crosscheck(ranking, support, original):
    import pytrec_eval

    qrels = {"q": {pid: 1 for pid in support}}
    surrogate = {pid: len(ranking) - i for i, pid in enumerate(ranking)}
    if sorted(surrogate, key=surrogate.get, reverse=True) != ranking:
        raise ValueError("rank surrogate changed input order")
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recall.2,4,6", "ndcg_cut.4", "recip_rank"})
    other = evaluator.evaluate({"q": surrogate})["q"]
    independent = {"ndcg@4": other["ndcg_cut_4"], "mrr": other["recip_rank"]}
    for k in (2, 4, 6):
        independent[f"recall@{k}"] = other[f"recall_{k}"]
        independent[f"all@{k}"] = int(set(support) <= set(ranking[:k]))
    return {k: abs(original[k] - v) for k, v in independent.items()}
