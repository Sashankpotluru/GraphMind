import {
  $,
  el,
  option,
  labels,
  datasets,
  percent,
  contextLabel,
} from "./dom.mjs";
import { context, data } from "./data.mjs";
import { EngineClient } from "./engine-client.mjs";
import {
  defaultFocus,
  neighborhood,
  rankData,
  renderMap,
  renderRanks,
} from "./graph-view.mjs";
import { showResults } from "./results-view.mjs";
import { bindParity } from "./parity.mjs";

const engine = new EngineClient();
const state = {
  ds: "hotpot",
  j: 0,
  mode: "recorded",
  method: "graph",
  depth: 2,
  c: null,
  live: null,
  focus: null,
  edge: null,
  mention: 0,
  support: false,
};
let revision = 0,
  catalog,
  index,
  activePage;
const methodName = () =>
  state.method === "graph" ? "graph-" + state.depth : state.method;
const isLive = () => state.mode === "custom";
function message(value) {
  $("workspace-status").textContent = value;
}
function firstFocus(ranks) {
  state.focus = defaultFocus(state.c.graph, ranks);
  const first = neighborhood(state.c.graph, state.focus)
    .slice()
    .sort((a, b) => {
      const ids = ranks.map((row) => row.id);
      const other = (e) => (e.source === state.focus ? e.target : e.source);
      return ids.indexOf(other(a)) - ids.indexOf(other(b));
    })[0];
  state.edge = first
    ? first.target === state.focus
      ? first.source
      : first.target
    : null;
  state.mention = 0;
}
function resetRun() {
  ++revision;
  engine.reset();
  state.live = null;
  $("run").disabled = false;
}
function renderCase(focusAfter) {
  $("sample-meta").textContent = contextLabel(state.c, state.mode);
  if (!state.c || (isLive() && !state.live)) {
    $("case-content").hidden = true;
    return;
  }
  const method = methodName(),
    c = state.c,
    ranks = rankData(c, method, state.live);
  if (!state.focus) firstFocus(ranks);
  $("case-content").hidden = false;
  $("method").disabled = isLive();
  $("depth").parentElement.hidden = isLive();
  $("depth").disabled = !isLive() && state.method !== "graph";
  $("support").disabled = isLive();
  $("support").checked = state.support;
  renderMap(state, ranks, renderCase);
  renderRanks(state, ranks, method, (id) => {
    state.focus = id;
    state.edge = null;
    state.mention = 0;
    renderCase("focus");
    $("map-head").scrollIntoView({ block: "start" });
  });
  $("reader").hidden = isLive();
  $("responses").replaceChildren();
  if (!isLive()) {
    for (const { method: readerMethod, input, prediction } of c.readers) {
      const box = el(
        "div",
        undefined,
        "answer" + (readerMethod === method ? " current" : ""),
      );
      const title = el("h3", labels[readerMethod]);
      if (readerMethod === method) title.append(el("small", "Selected"));
      const observed =
        prediction.answer === "" ? "Empty model response" : prediction.answer;
      box.append(
        title,
        el("p", observed, prediction.answer === "" ? "empty-response" : ""),
      );
      const stats = c.answerMetrics.methods[readerMethod];
      box.append(
        el(
          "small",
          "EM " + percent(stats.em) + " · token F1 " + percent(stats.f1),
        ),
      );
      const details = el("details");
      details.append(
        el("summary", "Inspect reader input"),
        el("pre", input.text),
      );
      box.append(details);
      $("responses").append(box);
    }
    if (!c.readers.some((row) => row.method === method))
      $("responses").prepend(
        el(
          "p",
          "The reader experiment covered Dense, Hybrid and Graph at depth 2. The current retrieval method has no recorded answer.",
          "notice",
        ),
      );
    if (state.support)
      $("responses").append(
        el(
          "p",
          "Reference answer: " + c.judgment.answers.join(" / "),
          "notice",
        ),
      );
  }
  if (focusAfter) $(focusAfter)?.focus({ preventScroll: true });
}
function sampleMenus() {
  const list = catalog[state.ds];
  $("sample").replaceChildren(
    ...list.map((x, i) =>
      option(
        i,
        String(i + 1).padStart(2, "0") +
          " / " +
          list.length +
          " — " +
          x.question,
      ),
    ),
  );
  $("sample").disabled = false;
  $("sample").value = String(state.j);
  $("prev").disabled = state.j === 0;
  $("next").disabled = state.j === list.length - 1;
}
function setMode(mode) {
  if (state.mode === mode) return;
  resetRun();
  message("");
  state.mode = mode;
  if (!isLive() && state.depth === 0) state.depth = 2;
  $("depth").value = String(state.depth);
  $("live-depth").value = String(state.depth);
  state.focus = null;
  state.support = false;
  $("recorded").setAttribute("aria-pressed", String(!isLive()));
  $("custom").setAttribute("aria-pressed", String(isLive()));
  $("live-form").hidden = !isLive();
  $("question").hidden = isLive();
  $("method").replaceChildren(
    ...(isLive()
      ? [option("graph", "BM25-seeded graph")]
      : [
          option("graph", "Graph expansion"),
          option("hybrid", "Hybrid RRF"),
          option("dense", "Dense · MiniLM"),
          option("bm25", "BM25"),
        ]),
  );
  state.method = "graph";
  $("method").value = "graph";
  renderCase();
  if (isLive()) $("query").focus();
}
async function loadCase(ds, j) {
  resetRun();
  const ticket = revision;
  Object.assign(state, {
    ds,
    j,
    c: null,
    focus: null,
    edge: null,
    mention: 0,
    support: false,
  });
  $("dataset").value = ds;
  $("sample").disabled = true;
  $("prev").disabled = true;
  $("next").disabled = true;
  $("query").value = "";
  $("case-content").hidden = true;
  $("retry").hidden = true;
  $("question").textContent = "Loading question…";
  $("sample-meta").textContent = "";
  message("Loading verified context…");
  try {
    [catalog, index] = await Promise.all([
      data("catalog.json"),
      data("results/explorer-index.json"),
    ]);
    if (ticket !== revision) return;
    sampleMenus();
    const c = await context(ds, j, index);
    if (ticket !== revision) return;
    state.c = c;
    $("question").textContent = c.example.question;
    message("");
    renderCase();
  } catch (error) {
    if (ticket !== revision) return;
    $("question").textContent = "Context unavailable";
    message(error.message);
    $("retry").hidden = false;
  }
}
function modeClick(mode) {
  if (state.mode === mode) return;
  const pending = !state.c;
  setMode(mode);
  if (pending) loadCase(state.ds, state.j);
}
async function run(event) {
  event?.preventDefault();
  if (!state.c) {
    message("Wait for the context to load.");
    return;
  }
  const question = $("query").value;
  resetRun();
  const ticket = revision;
  state.focus = null;
  state.edge = null;
  state.mention = 0;
  $("case-content").hidden = true;
  $("run").disabled = true;
  message("Running BM25 and graph expansion in the Python worker…");
  try {
    const result = await engine.run({
      documents: state.c.example.documents,
      question,
      depth: state.depth,
    });
    if (ticket !== revision) return;
    state.live = result;
    renderCase();
    message(
      "Live retrieval complete for “" +
        question +
        "”. " +
        result.ranking.length +
        " candidate paragraphs ranked.",
    );
  } catch (error) {
    if (ticket !== revision || error.name === "AbortError") return;
    message(error.message + " Edit your question or run retrieval again.");
  } finally {
    if (ticket === revision) $("run").disabled = false;
  }
}
$("dataset").onchange = () => loadCase($("dataset").value, 0);
$("sample").onchange = () => loadCase(state.ds, Number($("sample").value));
$("prev").onclick = () => loadCase(state.ds, state.j - 1);
$("next").onclick = () => loadCase(state.ds, state.j + 1);
$("retry").onclick = () => loadCase(state.ds, state.j);
$("recorded").onclick = () => modeClick("recorded");
$("custom").onclick = () => modeClick("custom");
$("live-form").onsubmit = run;
$("query").oninput = () => {
  if (state.c && (state.live || $("run").disabled)) {
    resetRun();
    message("");
    renderCase();
  }
};
$("method").onchange = () => {
  state.method = $("method").value;
  renderCase();
};
$("depth").onchange = () => {
  state.depth = Number($("depth").value);
  if (isLive()) run();
  else renderCase();
};
$("focus").onchange = () => {
  state.focus = $("focus").value;
  state.edge = null;
  state.mention = 0;
  renderCase("focus");
};
$("edge").onchange = () => {
  state.edge = $("edge").value || null;
  state.mention = 0;
  renderCase("edge");
};
$("live-depth").onchange = () => {
  state.depth = Number($("live-depth").value);
  if (state.live || $("run").disabled) run();
};
$("rank-filter").onchange = () => renderCase("rank-filter");
$("support").onchange = () => {
  state.support = $("support").checked;
  renderCase();
};
matchMedia("(max-width: 600px)").addEventListener("change", () => renderCase());

function navigate() {
  const page = ["results", "architecture"].includes(location.hash.slice(1))
    ? location.hash.slice(1)
    : "workspace";
  for (const name of ["workspace", "results", "architecture"])
    $(name).hidden = name !== page;
  document.querySelectorAll("nav a").forEach((a) => {
    if (a.hash === "#" + page) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  if (page === "results")
    showResults(async (ds, loss) => {
      const choices = (await data("catalog.json"))[ds];
      const i = loss ? choices.findIndex((q) => q.delta < 0) : 0;
      if (i < 0) return;
      modeClick("recorded");
      state.method = "graph";
      state.depth = 2;
      $("method").value = "graph";
      $("depth").value = "2";
      location.hash = "workspace";
      loadCase(ds, i);
    });
  if (activePage && activePage !== page) {
    window.scrollTo(0, 0);
    const h = $(page).querySelector("h1");
    h.tabIndex = -1;
    h.focus({ preventScroll: true });
  }
  activePage = page;
}
window.addEventListener("hashchange", navigate);
bindParity();
navigate();
loadCase("hotpot", 0);
