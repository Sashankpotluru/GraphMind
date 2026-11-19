import { bytes } from "./data.mjs";
import { validatePayload } from "./engine-client.mjs";

let ready;
async function initialize() {
  const runtime = new URL("./runtime/", import.meta.url);
  const [{ loadPyodide }, core] = await Promise.all([
    import(new URL("pyodide.mjs", runtime).href),
    bytes("engine.zip"),
  ]);
  const py = await loadPyodide({ indexURL: runtime.href });
  py.FS.writeFile("/home/pyodide/engine.zip", new Uint8Array(core));
  py.runPython(`import sys, json
sys.path.insert(0, "/home/pyodide/engine.zip")
from graphmind import retrieve
def run_question(payload):
    return json.dumps(retrieve(**json.loads(payload)), allow_nan=False)
`);
  return py.globals.get("run_question");
}

self.onmessage = async ({ data: { id, payload } }) => {
  try {
    validatePayload(payload);
    ready ??= initialize();
    const run = await ready;
    const result = JSON.parse(run(JSON.stringify(payload)));
    self.postMessage({ id, result });
  } catch (error) {
    ready = null;
    self.postMessage({ id, error: String(error.message || error) });
  }
};
