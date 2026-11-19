import { $, el, option, labels } from "./dom.mjs";

const ns = "http://www.w3.org/2000/svg";
function svg(tag, attrs, text) {
  const node = document.createElementNS(ns, tag);
  Object.entries(attrs).forEach(([name, value]) =>
    node.setAttribute(name, value),
  );
  if (text !== undefined) node.textContent = text;
  return node;
}
function action(node, label, handler) {
  node.setAttribute("role", "button");
  node.setAttribute("tabindex", "0");
  node.setAttribute("aria-label", label);
  node.append(svg("title", {}, label));
  node.addEventListener("click", handler);
  node.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      handler();
    }
  });
}
export function neighborhood(graph, id) {
  return graph.projection.filter((e) => e.source === id || e.target === id);
}
export function defaultFocus(graph, ranks) {
  return ranks.reduce(
    (best, row) =>
      neighborhood(graph, row.id).length > neighborhood(graph, best.id).length
        ? row
        : best,
    ranks[0],
  ).id;
}
export function rankData(c, method, live) {
  if (live) return live.ranking;
  const g = c.run.graph[method];
  return c.run.rankings[method].map((i) => ({
    id: c.example.documents[i].id,
    score: g ? g.score[i] : c.run.scores[method][i],
    ...(g
      ? { base_score: g.base_score[i], graph_score: g.graph_score[i] }
      : {}),
  }));
}
function titleLines(title, limit) {
  const words = title.trim().split(/\s+/);
  const lines = [""];
  for (const word of words) {
    const i = lines.length - 1;
    if (lines[i] && (lines[i] + " " + word).length > limit) lines.push(word);
    else lines[i] += (lines[i] ? " " : "") + word;
  }
  const shown = lines.slice(0, 2);
  const last = shown.length - 1;
  if (shown[last].length > limit || lines.length > 2)
    shown[last] = shown[last].slice(0, limit - 1) + "…";
  return shown;
}

function locations(count, small) {
  const cx = small ? 175 : 345,
    cy = small ? 151 : 142;
  const offsets = small
    ? [
        [-100, -78],
        [101, -77],
        [101, 75],
        [-100, 75],
      ]
    : [
        [-199, -61],
        [192, -71],
        [169, 77],
        [-185, 79],
        [-10, -94],
        [6, 103],
      ];
  if (count === 1)
    return [
      [cx, cy],
      [cx + (small ? 98 : 194), cy - 52],
    ];
  return [
    [cx, cy],
    ...offsets.slice(0, count).map(([x, y]) => [cx + x, cy + y]),
  ];
}
function drawNode(plot, doc, point, rank, main, supported, small, handler) {
  const [x, y] = point;
  const g = svg("g", {
    class: "graph-node " + (main ? "hub" : "satellite"),
    transform: "translate(" + x + " " + y + ")",
  });
  action(g, "Inspect " + doc.title + ", rank " + rank, handler);
  if (main) g.append(svg("circle", { r: 36, class: "halo" }));
  g.append(svg("circle", { r: main ? 27 : 17, class: "node-body" }));
  if (main)
    g.append(
      svg("path", {
        d: "M-12 -9 Q-5 -12 0 -7 Q5 -12 12 -9 V8 Q6 5 0 10 Q-6 5 -12 8 Z M0 -7 V10",
        class: "book",
      }),
    );
  else
    g.append(
      svg(
        "text",
        { y: 4, class: "rank-num", "text-anchor": "middle" },
        String(rank),
      ),
    );
  const lines = titleLines(doc.title, small ? 15 : 24);
  lines.forEach((line, i) =>
    g.append(
      svg(
        "text",
        {
          y: (main ? 51 : 33) + i * 15,
          "text-anchor": "middle",
          class: "node-label",
        },
        line,
      ),
    ),
  );
  if (supported)
    g.append(
      svg(
        "text",
        {
          y: (main ? 51 : 33) + lines.length * 15,
          "text-anchor": "middle",
          class: "support-label",
        },
        "Gold support",
      ),
    );
  plot.append(g);
}
function sourceText(text, mention) {
  const excerpt = el("blockquote");
  if (!mention) {
    excerpt.textContent = text.slice(0, 290) + (text.length > 290 ? "…" : "");
    return excerpt;
  }
  const chars = Array.from(text),
    start = Math.max(0, mention.start - 84),
    end = Math.min(chars.length, mention.end + 104);
  excerpt.append(
    (start ? "…" : "") + chars.slice(start, mention.start).join(""),
  );
  excerpt.append(el("mark", chars.slice(mention.start, mention.end).join("")));
  excerpt.append(
    chars.slice(mention.end, end).join("") + (end < chars.length ? "…" : ""),
  );
  return excerpt;
}

function inspect(state, ranks, connection, rerender) {
  const { c, focus, support } = state,
    panel = $("inspection");
  const docs = new Map(c.graph.paragraphs.map((d) => [d.id, d]));
  const entities = new Map(c.graph.entities.map((e) => [e.id, e]));
  const mentions = c.graph.mentions.filter((m) =>
    connection
      ? connection.evidence_edge_ids.includes(m.id)
      : m.paragraph_id === focus,
  );
  const mention = mentions[state.mention] ?? mentions[0];
  const doc = docs.get(mention?.paragraph_id ?? focus);
  panel.replaceChildren();
  const chain = connection ? "Connection evidence" : "Paragraph evidence";
  panel.append(el("p", chain, "category"), el("h3", doc.title));
  if (support && c.judgment.support_ids.includes(doc.id))
    panel.append(el("p", "Annotated supporting paragraph", "support-note"));
  if (mention) {
    const entity = entities.get(mention.entity_id);
    const origin = el("p", undefined, "relation");
    origin.append("Mentions ", el("strong", entity.title));
    panel.append(
      origin,
      sourceText(doc.text, mention),
      el(
        "p",
        "Source span [" +
          mention.start +
          ", " +
          mention.end +
          ") · Unicode code points",
        "coord",
      ),
    );
    if (mentions.length > 1) {
      const field = el("label", "Mention");
      const select = el("select");
      mentions.forEach((m, i) =>
        select.add(option(i, m.text + " [" + m.start + ", " + m.end + ")")),
      );
      select.value = String(mentions.indexOf(mention));
      select.onchange = () => {
        state.mention = Number(select.value);
        rerender("mention");
      };
      select.id = "mention";
      field.append(select);
      panel.append(field);
    }
  } else {
    panel.append(
      el(
        "p",
        connection
          ? "Both paragraphs own the same canonical title entity."
          : "This paragraph has no extracted title mentions.",
        "relation",
      ),
      sourceText(doc.text),
    );
  }
  const full = el("details"),
    summary = el("summary", "Read full source paragraph");
  full.append(summary, el("p", doc.text, "full-text"));
  panel.append(full);
  const rank = ranks.findIndex((r) => r.id === doc.id) + 1;
  panel.append(
    el(
      "p",
      "Source paragraph ranked " + rank + " of " + ranks.length + ".",
      "coord",
    ),
  );
}
export function renderMap(state, ranks, rerender) {
  const { c, focus, support } = state;
  const graph = c.graph,
    docs = new Map(graph.paragraphs.map((d) => [d.id, d]));
  const ids = ranks.map((r) => r.id),
    edges = neighborhood(graph, focus);
  const other = (e) => (e.source === focus ? e.target : e.source);
  edges.sort((a, b) => ids.indexOf(other(a)) - ids.indexOf(other(b)));
  const connection = edges.find((e) => other(e) === state.edge);
  const small = matchMedia("(max-width: 600px)").matches;
  const plot = $("plot"),
    count = small ? 4 : 6;
  const display = edges.slice(0, count);
  if (connection && !display.includes(connection))
    display[display.length - 1] = connection;
  plot.replaceChildren();
  plot.setAttribute("viewBox", small ? "0 0 350 318" : "0 0 690 316");
  const points = locations(display.length, small);
  const [cx, cy] = points[0];
  const defs = svg("defs", {}),
    pattern = svg("pattern", {
      id: "speck",
      width: 16,
      height: 16,
      patternUnits: "userSpaceOnUse",
    });
  pattern.append(svg("circle", { cx: 2, cy: 2, r: 0.8, fill: "#c6bacd" }));
  defs.append(pattern);
  plot.append(defs);
  plot.append(
    svg("rect", { width: "100%", height: "100%", fill: "url(#speck)" }),
  );
  for (const radius of [71, 121, 177])
    plot.append(
      svg("ellipse", {
        cx,
        cy,
        rx: radius * (small ? 1 : 1.31),
        ry: radius * 0.78,
        class: "orbit",
      }),
    );
  display.forEach((e, i) => {
    const [x, y] = points[i + 1];
    const line = svg("g", {
      class: "graph-edge" + (connection === e ? " chosen" : ""),
    });
    const pick = () => {
      state.edge = other(e);
      state.mention = 0;
      rerender("edge");
    };
    action(line, "Inspect connection to " + docs.get(other(e)).title, pick);
    const attrs = { x1: cx, y1: cy, x2: x, y2: y };
    line.append(
      svg("line", { ...attrs, class: "edge-hit" }),
      svg("line", { ...attrs, class: "edge-line" }),
    );
    const mx = (cx + x) / 2,
      my = (cy + y) / 2;
    line.append(
      svg("rect", {
        x: mx - 3,
        y: my - 3,
        width: 6,
        height: 6,
        transform: "rotate(45 " + mx + " " + my + ")",
        class: "edge-mark",
      }),
    );
    plot.append(line);
  });
  [focus, ...display.map(other)].forEach((id, i) => {
    drawNode(
      plot,
      docs.get(id),
      points[i],
      ids.indexOf(id) + 1,
      i === 0,
      support && c.judgment.support_ids.includes(id),
      small,
      () => {
        state.focus = id;
        state.edge = null;
        state.mention = 0;
        rerender("focus");
      },
    );
  });
  const stat = edges.length
    ? display.length + " of " + edges.length + " neighboring paragraphs shown"
    : "No linked neighbors. Choose another paragraph below.";
  $("map-stats").textContent = stat;
  $("focus").replaceChildren(
    ...ids.map((id, i) => option(id, i + 1 + ". " + docs.get(id).title)),
  );
  $("focus").value = focus;
  $("edge").replaceChildren(
    option("", "Paragraph only"),
    ...edges.map((e) => option(other(e), docs.get(other(e)).title)),
  );
  $("edge").value = state.edge || "";
  $("edge").disabled = !edges.length;
  inspect(state, ranks, connection, rerender);
}
export function renderRanks(state, ranks, method, onPick) {
  const { c, live, support } = state,
    panel = $("rank-list");
  const docs = new Map(c.example.documents.map((d) => [d.id, d]));
  const hybrid = c.run.rankings.hybrid.map((i) => c.example.documents[i].id);
  $("rank-summary").textContent = live
    ? "Live BM25 seeds · normalized scores after expansion."
    : labels[method] +
      " · hybrid rank shown for comparison. Scores use each method’s own scale.";
  const filter = $("rank-filter");
  filter.querySelector('[value="gold"]').disabled = !support;
  if (!support && filter.value === "gold") filter.value = "top";
  if (support) {
    const gold = new Set(c.judgment.support_ids);
    const found = (ids) => ids.slice(0, 4).filter((id) => gold.has(id)).length;
    $("rank-summary").append(
      " Top four contain " +
        found(ranks.map((r) => r.id)) +
        " of " +
        gold.size +
        " supporting paragraphs. Hybrid contains " +
        found(hybrid) +
        " of " +
        gold.size +
        ".",
    );
  }
  panel.replaceChildren();
  const list = el("ol", undefined, "ranks");
  ranks.forEach((row, i) => {
    if (filter.value === "top" && i >= 4) return;
    if (filter.value === "gold" && !c.judgment.support_ids.includes(row.id))
      return;
    const item = el("li"),
      b = el("button");
    b.type = "button";
    b.onclick = () => onPick(row.id);
    b.append(
      el("span", String(i + 1), "rank-index"),
      el("span", docs.get(row.id).title, "rank-title"),
      el("span", row.score.toFixed(4), "score"),
    );
    item.append(b);
    const note = [];
    if (!live) note.push("Hybrid #" + (hybrid.indexOf(row.id) + 1));
    if ("base_score" in row)
      note.push(
        "Base " + row.base_score.toFixed(4),
        "walk " + row.graph_score.toFixed(4),
      );
    if (support && c.judgment.support_ids.includes(row.id))
      note.push("Gold support");
    if (note.length) item.append(el("p", note.join(" · ")));
    list.append(item);
  });
  panel.append(list);
}
