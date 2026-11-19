"""Question-paired inference and support-connected component sensitivity analysis."""


def components(supports):
    parent = list(range(len(supports)))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, support in enumerate(supports):
        for pid in support:
            other = owner.setdefault(pid, i)
            parent[root(other)] = root(i)
    groups = {}
    for i in range(len(parent)):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def bootstrap(totals, counts):
    import numpy as np

    rng = np.random.default_rng(20261119)
    n = len(counts)
    samples = []
    totals = np.asarray(totals, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    for _ in range(100):
        indices = rng.integers(n, size=(100, n))
        samples.extend(totals[indices].sum(axis=1) / counts[indices].sum(axis=1))
    return np.quantile(samples, [0.025, 0.975], method="linear").tolist()


def summarize(deltas, groups):
    from scipy.stats import binomtest

    wins = deltas.count(1)
    losses = deltas.count(-1)
    sizes = [len(g) for g in groups]
    return {
        "n": len(deltas),
        "wins": wins,
        "losses": losses,
        "ties": deltas.count(0),
        "delta": sum(deltas) / len(deltas),
        "p": float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0,
        "query_ci95": bootstrap(deltas, [1] * len(deltas)),
        "cluster_ci95": bootstrap([sum(deltas[i] for i in g) for g in groups], sizes)
        if len(groups) > 1
        else None,
        "clusters": len(groups),
        "cluster_sizes": sorted(sizes, reverse=True),
        "largest_cluster_share": max(sizes) / len(deltas),
    }


def holm(p_values):
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for i, j in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, min(1.0, (len(p_values) - i) * p_values[j]))
        adjusted[j] = running
    return adjusted
