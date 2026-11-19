import assert from "node:assert/strict";
import { test } from "node:test";
import { contextLabel } from "../dom.mjs";

test("question types belong only to the selected recorded question", () => {
  const hotpot = {
    example: { documents: Array(10) },
    judgment: { stratum: "bridge" },
  };
  const musique = {
    example: { documents: Array(20) },
    judgment: { stratum: "4hop" },
  };
  assert.equal(
    contextLabel(hotpot, "recorded"),
    "bridge question · 10 candidate paragraphs",
  );
  assert.equal(contextLabel(hotpot, "custom"), "10 candidate paragraphs");
  assert.equal(contextLabel(null, "custom"), "");
  assert.equal(contextLabel(musique, "custom"), "20 candidate paragraphs");
  assert.equal(
    contextLabel(musique, "recorded"),
    "4-hop question · 20 candidate paragraphs",
  );
});
