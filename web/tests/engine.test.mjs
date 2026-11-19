import { test } from "node:test";
import assert from "node:assert/strict";
import { EngineClient, validatePayload } from "../engine-client.mjs";

function harness() {
  const workers = [];
  const client = new EngineClient(() => {
    const worker = {
      postMessage(message) {
        this.last = message;
      },
      terminate() {
        this.dead = true;
      },
      reply(result, id = this.last.id) {
        this.onmessage?.({ data: { id, result } });
      },
    };
    workers.push(worker);
    return worker;
  });
  return { client, workers };
}
const payload = {
  documents: [{ id: "p", title: "A", text: "📚 Café" }],
  question: "Café?",
  depth: 2,
};
test("cancelling a context change rejects the pending request and late old-worker replies cannot bind to the next context", async () => {
  const { client, workers } = harness();
  const first = client.run(payload);
  const rejected = assert.rejects(first, { name: "AbortError" });
  client.cancelPending();
  const second = client.run({
    ...payload,
    documents: [{ id: "b", title: "Other", text: "Another context" }],
  });
  workers[0].reply("stale");
  workers[1].reply("fresh");
  assert.equal(await second, "fresh");
  await rejected;
  assert.equal(workers[0].dead, true);
});
test("edits and successive queries reuse a ready worker", async () => {
  const { client, workers } = harness();
  const first = client.run(payload);
  workers[0].reply("first");
  assert.equal(await first, "first");
  client.cancelPending(); // Edit after displaying the first result.
  const next = client.run({ ...payload, question: "Second question?" });
  assert.equal(workers.length, 1);
  assert.equal(workers[0].last.payload.question, "Second question?");
  workers[0].reply("second");
  assert.equal(await next, "second");
  assert.equal(workers[0].dead, undefined);
});
test("an initialization error rejects, terminates, and retries using a fresh worker", async () => {
  const { client, workers } = harness();
  const first = client.run(payload);
  workers[0].onmessage({
    data: { id: workers[0].last.id, error: "cannot initialize" },
  });
  await assert.rejects(first, /cannot initialize/);
  const next = client.run(payload);
  workers[1].reply("retried");
  assert.equal(await next, "retried");
});
test("a second request supersedes the first even while initializing", async () => {
  const { client, workers } = harness();
  const first = client.run(payload);
  const rejected = assert.rejects(first, { name: "AbortError" });
  const second = client.run({ ...payload, question: "New" });
  workers[0].reply("old");
  workers[1].reply("new");
  assert.equal(await second, "new");
  await rejected;
});
test("worker payload has a bounded data-only context and counts Unicode code points", () => {
  assert.doesNotThrow(() =>
    validatePayload({ ...payload, question: "📚".repeat(2000) }),
  );
  for (const field of [
    { question: "  " },
    { question: "📚".repeat(2001) },
    { depth: 4 },
    { documents: [] },
    { documents: [{ ...payload.documents[0], text: "x".repeat(50_001) }] },
    { documents: [{ ...payload.documents[0], support: true }] },
  ])
    assert.throws(() => validatePayload({ ...payload, ...field }));
});
