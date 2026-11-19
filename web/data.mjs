const base = new URL("./", import.meta.url);
let indexPromise;
const cache = new Map();

function safePath(path) {
  if (
    !/^[A-Za-z0-9_./-]+$/.test(path) ||
    path.startsWith("/") ||
    path.split("/").some((part) => !part || part === "." || part === "..")
  )
    throw new Error("Unrecognized artifact path.");
}

async function response(path) {
  safePath(path);
  const res = await fetch(new URL(path, base));
  if (!res.ok)
    throw new Error("Could not load " + path + " (HTTP " + res.status + ").");
  return res;
}

async function manifest() {
  indexPromise ??= response("site-manifest.json")
    .then((r) => r.json())
    .then((data) => new Map(data.files.map((row) => [row.path, row])))
    .catch((error) => {
      indexPromise = null;
      throw error;
    });
  return indexPromise;
}

export async function bytes(path) {
  safePath(path);
  const row = (await manifest()).get(path);
  if (!row) throw new Error("Missing artifact in site inventory: " + path);
  const raw = await (await response(path)).arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", raw);
  const hex = Array.from(new Uint8Array(digest), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
  if (row.bytes !== raw.byteLength || row.sha256 !== hex)
    throw new Error("Artifact failed verification: " + path);
  return raw;
}

async function parse(path) {
  const raw = await bytes(path);
  if (path.endsWith(".jsonl.gz")) {
    const stream = new Blob([raw])
      .stream()
      .pipeThrough(new DecompressionStream("gzip"));
    const text = await new Response(stream).text();
    return text
      .trimEnd()
      .split("\n")
      .map((line) => JSON.parse(line));
  }
  return JSON.parse(new TextDecoder().decode(raw));
}

export function data(path) {
  if (cache.has(path)) return cache.get(path);
  const request = parse(path).catch((error) => {
    if (cache.get(path) === request) cache.delete(path);
    throw error;
  });
  if (cache.size >= 36) cache.delete(cache.keys().next().value);
  cache.set(path, request);
  return request;
}

async function row(folder, index, size, qid, method) {
  const file = String(Math.floor(index / size)).padStart(4, "0");
  const record = (await data("results/" + folder + "/" + file + ".jsonl.gz"))[
    index % size
  ];
  if (record?.qid !== qid || (method && record.method !== method))
    throw new Error("Record identity mismatch.");
  return record;
}

export async function context(ds, j, index) {
  const dataset = index.datasets[ds];
  const qid = dataset.answer_ids[j],
    i = dataset.dev_ids.indexOf(qid);
  if (!qid || i < 0) throw new Error("Question is missing from the dev split.");
  const dev = "retrieval/" + ds + "/dev/",
    answers = "answers/" + ds + "/";
  const [example, judgment, run, metrics, { graph }, answerMetrics, readers] =
    await Promise.all([
      ...["cases", "judgments", "runs", "metrics"].map((name) =>
        row(dev + name, i, index.retrieval_shard_size, qid),
      ),
      row("explorer/" + ds, j, index.graph_shard_size, qid),
      row(answers + "metrics", j, index.answer_shard_size, qid),
      Promise.all(
        index.methods.map(async (method, k) => {
          const n = j * index.methods.length + k;
          const [input, prediction] = await Promise.all(
            ["inputs", "predictions"].map((name) =>
              row(answers + name, n, index.answer_shard_size, qid, method),
            ),
          );
          return { method, input, prediction };
        }),
      ),
    ]);
  return { example, judgment, run, metrics, graph, answerMetrics, readers };
}
