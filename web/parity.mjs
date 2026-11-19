import { data } from "./data.mjs";
import { EngineClient } from "./engine-client.mjs";
import { $ } from "./dom.mjs";

function equal(actual, expected) {
  if (typeof expected === "number")
    return Number.isFinite(actual) && Math.abs(actual - expected) < 1e-12;
  if (!expected || typeof expected !== "object") return actual === expected;
  if (
    !actual ||
    typeof actual !== "object" ||
    Array.isArray(actual) !== Array.isArray(expected)
  )
    return false;
  const keys = Object.keys(expected);
  return (
    keys.length === Object.keys(actual).length &&
    keys.every((k) => Object.hasOwn(actual, k) && equal(actual[k], expected[k]))
  );
}
export function bindParity() {
  $("parity").onclick = async () => {
    const client = new EngineClient(),
      status = $("parity-status");
    $("parity").disabled = true;
    status.textContent = "Starting Python in a worker…";
    const passed = [];
    try {
      for (const { name, payload, expected } of await data("parity.json")) {
        const result = await client.run(payload);
        for (const field of Object.keys(expected)) {
          if (!equal(result[field], expected[field]))
            throw new Error(name + " differs in " + field + ".");
        }
        passed.push(name + " (" + result.ranking.length + " paragraphs)");
      }
      status.textContent =
        "Both fixtures matched native Python. " + passed.join(" and ") + ".";
    } catch (error) {
      status.textContent =
        "Check failed. " + error.message + " Run again to retry.";
    } finally {
      client.reset();
      $("parity").disabled = false;
    }
  };
}
