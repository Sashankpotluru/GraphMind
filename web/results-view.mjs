import { $, el, labels, datasets, button, percent, cellTable } from "./dom.mjs";
import { data } from "./data.mjs";

const keys = [
  "all@2",
  "all@4",
  "all@6",
  "recall@2",
  "recall@4",
  "recall@6",
  "ndcg@4",
  "mrr",
];
const pp = (value) => (100 * value).toFixed(2);
const signed = (value) => (value >= 0 ? "+" : "") + pp(value);
const interval = (arr) =>
  arr ? arr.map(pp).join(" to ") + " pp" : "unavailable";
let ticket = 0;
export async function showResults(openSample, ds = "hotpot") {
  const token = ++ticket,
    panel = $("results-content");
  panel.textContent = "Loading measured results…";
  try {
    const dev = "results/retrieval/" + ds + "/dev/";
    const [aggregate, primary, answers, times, catalog] = await Promise.all([
      data(dev + "aggregate.json"),
      data("results/retrieval/primary.json"),
      data("results/answers/summary.json"),
      data(dev + "timing-summary.json"),
      data("catalog.json"),
    ]);
    if (ticket !== token) return;
    panel.replaceChildren();
    const nav = el("div", undefined, "mode-switch result-tabs");
    for (const [key, label] of Object.entries(datasets)) {
      const b = button(label, () => showResults(openSample, key));
      b.setAttribute("aria-pressed", String(key === ds));
      nav.append(b);
    }
    panel.append(nav);
    const main = el("section", undefined, "result-section"),
      r = primary[ds],
      a = answers[ds];
    const g = aggregate.overall.methods,
      p = a.primary_graph2_minus_hybrid_f1;
    main.append(
      el("h2", "Complete evidence in the first four ranks"),
      el(
        "p",
        "AllSupport@4 counts a question once when the first four results contain every gold paragraph. On " +
          datasets[ds] +
          ", graph depth 2 changed that rate by " +
          signed(r.delta) +
          " percentage points across " +
          r.n.toLocaleString("en-US") +
          " dev questions.",
      ),
    );
    const chart = el("div", undefined, "bar-chart");
    for (const [key, label] of Object.entries(labels)) {
      const row = el("div", undefined, "bar-row"),
        track = el("div", undefined, "track"),
        fill = el(
          "div",
          undefined,
          key === "graph-2" ? "fill main-fill" : "fill",
        );
      fill.style.width = percent(g[key]["all@4"]);
      track.append(fill);
      row.append(
        el("span", label),
        track,
        el("strong", percent(g[key]["all@4"])),
      );
      chart.append(row);
    }
    main.append(
      chart,
      el(
        "p",
        "Paired outcomes: " +
          r.wins +
          " wins, " +
          r.losses +
          " losses and " +
          r.ties +
          " ties against Hybrid. The 95% query bootstrap interval is " +
          interval(r.query_ci95) +
          ". Resampling shared-support components gives " +
          interval(r.cluster_ci95) +
          " (" +
          r.clusters.toLocaleString("en-US") +
          " components, largest " +
          r.cluster_sizes[0] +
          "). Holm-adjusted sign-flip p = " +
          r.p_holm.toExponential(2) +
          ".",
      ),
    );
    main.append(
      el(
        "p",
        "The first relevant paragraph moved down on average. Hybrid MRR was " +
          g.hybrid.mrr.toFixed(3) +
          " and graph depth 2 reached " +
          g["graph-2"].mrr.toFixed(3) +
          ". Additional graph depth did not consistently improve evidence coverage.",
      ),
    );
    const controls = el("div", undefined, "example-controls");
    controls.append(
      button("Inspect first sampled question", () => openSample(ds, false)),
    );
    if (catalog[ds].some((q) => q.delta < 0))
      controls.append(
        button("Inspect a retrieval loss", () => openSample(ds, true)),
      );
    main.append(controls);
    panel.append(main);
    const all = el("section", undefined, "result-section");
    all.append(
      el("h2", "Retrieval across cutoffs"),
      el(
        "p",
        "Means across the entire development split. Each metric is shown on a 0–1 scale.",
      ),
      cellTable(
        [
          "Method",
          "All@2",
          "All@4",
          "All@6",
          "Recall@2",
          "Recall@4",
          "Recall@6",
          "nDCG@4",
          "MRR",
        ],
        Object.entries(labels).map(([k, l]) => [
          l,
          ...keys.map((m) => g[k][m].toFixed(4)),
        ]),
        "All methods and metrics",
      ),
    );
    const strata = Object.entries(aggregate).filter(([k]) => k !== "overall");
    all.append(
      el("h3", "By question type"),
      el(
        "p",
        "Dataset question types are annotations. Graph depth counts paragraph-to-paragraph transitions.",
      ),
      cellTable(
        [
          "Question type",
          "Questions",
          "Hybrid All@4",
          "Graph-2 All@4",
          "Difference (pp)",
        ],
        strata.map(([k, s]) => [
          k,
          s.n,
          percent(s.methods.hybrid["all@4"]),
          percent(s.methods["graph-2"]["all@4"]),
          signed(s.methods["graph-2"]["all@4"] - s.methods.hybrid["all@4"]),
        ]),
        "Descriptive strata",
      ),
    );
    panel.append(all);
    const reader = el("section", undefined, "result-section");
    reader.append(
      el("h2", "Did better retrieval improve answers?"),
      el(
        "p",
        "On 100 hash-selected questions, the graph-2 minus hybrid difference in FLAN-T5-small token F1 was " +
          signed(p.delta) +
          " points. The 95% query interval spans " +
          interval(p.query_ci95) +
          " (Holm p = " +
          p.p_holm.toFixed(3) +
          "). This sample leaves the direction of the answer effect uncertain.",
      ),
      cellTable(
        ["Recorded retrieval", "Exact match", "Token F1"],
        Object.entries(a.methods).map(([k, v]) => [
          labels[k],
          percent(v.em),
          percent(v.f1),
        ]),
        "Matched reader scores",
      ),
      el(
        "p",
        "The reader saw four passages capped at 96 context tokens each. EM and F1 measure reference-string overlap. Model exposure to Wikipedia or benchmark questions during pretraining is unknown.",
      ),
    );
    panel.append(reader);
    const timing = el("section", undefined, "result-section");
    timing.append(
      el("h2", "Measured latency"),
      el(
        "p",
        "Warm CPU query latency on one Apple M4 Pro. Each of 64 questions had one warmup and five sequential measurements. Graph and dense-index setup are timed separately.",
      ),
      cellTable(
        ["Step", "p50 (ms)", "p95 (ms)"],
        [
          ...Object.entries(times.query).map(([key, v]) => [labels[key], v]),
          ...Object.entries(times.setup).map(([key, v]) => [
            key === "graph" ? "Graph setup" : "Dense index setup",
            v,
          ]),
        ].map(([k, v]) => [
          k,
          (v.p50_ns / 1e6).toFixed(2),
          (v.p95_ns / 1e6).toFixed(2),
        ]),
        "Recorded query and setup times",
      ),
    );
    panel.append(timing);
  } catch (error) {
    if (ticket !== token) return;
    panel.replaceChildren(
      el("p", error.message),
      button("Retry results", () => showResults(openSample, ds)),
    );
  }
}
