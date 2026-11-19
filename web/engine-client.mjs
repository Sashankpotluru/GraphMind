export function validatePayload({ documents, question, depth }) {
  if (
    typeof question !== "string" ||
    !question.trim() ||
    [...question].length > 2000
  )
    throw new Error("Enter a question between 1 and 2,000 characters.");
  if (!Number.isInteger(depth) || depth < 0 || depth > 3)
    throw new Error("Choose a depth from 0 to 3.");
  if (
    !Array.isArray(documents) ||
    documents.length < 1 ||
    documents.length > 100
  )
    throw new Error("This context needs 1–100 paragraphs.");
  const seen = new Set();
  let chars = 0;
  for (const doc of documents) {
    if (
      !doc ||
      Object.keys(doc).sort().join() !== "id,text,title" ||
      typeof doc.id !== "string" ||
      !doc.id.trim() ||
      seen.has(doc.id) ||
      typeof doc.title !== "string" ||
      !doc.title.trim() ||
      typeof doc.text !== "string"
    )
      throw new Error("Invalid paragraph fields.");
    const length = [...doc.text].length;
    chars += length;
    if (length > 50000 || chars > 1000000)
      throw new Error("Context exceeds the paragraph size limit.");
    seen.add(doc.id);
  }
}

export class EngineClient {
  constructor(
    create = () =>
      new Worker(new URL("./engine-worker.mjs", import.meta.url), {
        type: "module",
      }),
  ) {
    this.create = create;
    this.serial = 0;
  }
  reset() {
    this.worker?.terminate();
    this.worker = null;
    if (this.pending) {
      this.pending.reject(
        new DOMException("Request superseded.", "AbortError"),
      );
      this.pending = null;
    }
  }
  cancelPending() {
    if (this.pending) this.reset();
  }
  async run(payload) {
    this.cancelPending();
    validatePayload(payload);
    const worker = (this.worker ??= this.create());
    const id = ++this.serial;
    return new Promise((resolve, reject) => {
      this.pending = { id, reject };
      const finish = (error, result) => {
        if (worker !== this.worker || this.pending?.id !== id) return;
        this.pending = null;
        if (error) {
          this.reset();
          reject(new Error(error));
        } else resolve(result);
      };
      worker.onerror = (event) =>
        finish(event.message || "The Python worker could not start.");
      worker.onmessageerror = () =>
        finish("Could not read the Python response.");
      worker.onmessage = ({ data }) => {
        if (data?.id === id) finish(data.error, data.result);
      };
      try {
        worker.postMessage({ id, payload });
      } catch (error) {
        finish(error.message);
      }
    });
  }
}
