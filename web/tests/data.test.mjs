import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { context, data } from "../data.mjs";

const release = new URL("../../results/", import.meta.url);
const manifest = JSON.parse(await readFile(new URL("release.json", release)));
const files = manifest.files.map((row) => ({
  ...row,
  path: "results/" + row.path,
}));
const retry = Buffer.from('{"data":"<script>literal text</script>"}');
files.push({
  path: "retry.json",
  bytes: retry.length,
  sha256: createHash("sha256").update(retry).digest("hex"),
});
let damaged = true;
globalThis.fetch = async (url) => {
  const name = url.pathname.split("/web/").at(-1);
  if (name === "site-manifest.json") return Response.json({ files });
  if (name === "retry.json") {
    const bytes = damaged ? Buffer.from("corrupt") : retry;
    damaged = false;
    return new Response(bytes);
  }
  return new Response(await readFile(new URL(name.slice(8), release)));
};

test("browser joins all 200 released contexts to their 600 actual prompts and predictions", async () => {
  const index = await data("results/explorer-index.json");
  let empty = 0,
    predictions = 0;
  for (const [ds, ids] of Object.entries(index.datasets)) {
    for (const [j, qid] of ids.answer_ids.entries()) {
      const c = await context(ds, j, index);
      assert.equal(c.example.qid, qid);
      assert.deepEqual(c.graph.paragraphs, c.example.documents);
      for (const { method, input, prediction } of c.readers) {
        assert.deepEqual(
          input.passages.map((p) => p.id),
          c.run.rankings[method]
            .slice(0, 4)
            .map((i) => c.example.documents[i].id),
        );
        assert.equal(prediction.qid, qid);
        assert.equal(prediction.method, method);
        if (prediction.answer === "") empty++;
        predictions++;
      }
    }
  }
  assert.equal(predictions, 600);
  assert.equal(empty, 2);
});

test("a corrupt artifact fails and retry recovers the original literal dataset string", async () => {
  await assert.rejects(data("retry.json"), /verification/);
  assert.equal(
    (await data("retry.json")).data,
    "<script>literal text</script>",
  );
  await assert.rejects(data("../outside.json"), /Unrecognized/);
});
