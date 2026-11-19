export const $ = (id) => document.getElementById(id);
export function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
export function option(value, text) {
  return new Option(text, value);
}
export function button(text, handler, className) {
  const node = el("button", text, className);
  node.type = "button";
  node.addEventListener("click", handler);
  return node;
}
export const labels = {
  bm25: "BM25",
  dense: "Dense",
  hybrid: "Hybrid",
  "graph-1": "Graph · 1 hop",
  "graph-2": "Graph · 2 hops",
  "graph-3": "Graph · 3 hops",
};
export const datasets = { hotpot: "HotpotQA", musique: "MuSiQue" };
export function contextLabel(context, mode) {
  if (!context) return "";
  const count = context.example.documents.length + " candidate paragraphs";
  if (mode === "custom") return count;
  const type = context.judgment.stratum.replace(/^(\d)hop$/, "$1-hop");
  return type + " question · " + count;
}
export const percent = (x) => (100 * x).toFixed(2) + "%";
export function cellTable(headings, records, caption) {
  const wrap = el("div", undefined, "table-scroll");
  wrap.tabIndex = 0;
  wrap.setAttribute("role", "region");
  wrap.setAttribute("aria-label", caption);
  const table = el("table");
  table.append(el("caption", caption, "sr-only"));
  const header = table.createTHead().insertRow();
  for (const heading of headings) header.append(el("th", heading));
  const body = table.createTBody();
  for (const record of records) {
    const row = body.insertRow();
    record.forEach((value, index) =>
      row.append(el(index ? "td" : "th", value)),
    );
  }
  wrap.append(table);
  return wrap;
}
