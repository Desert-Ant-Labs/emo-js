import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { load } from "../src/index.node.js";

// Offline: load from local files so tests don't need network or a token.
const fixtures = fileURLToPath(new URL("../data", import.meta.url));
const model = await load({ localModelPath: fixtures, allowRemote: false });
const top = (text: string) => model.suggestions(text, 1)[0]?.emoji ?? "";

test("english predictions", () => {
  assert.ok(["💰", "💳", "🧾", "🏦"].includes(top("Pay my bills")));
  assert.ok(["🐕", "🐾", "🚶"].includes(top("walk the dog")));
  assert.equal(top("book a flight to Tokyo"), "✈️");
  assert.ok(["🦷", "📅", "🏥"].includes(top("dentist appointment")));
});

test("multilingual predictions", () => {
  assert.ok(["🐕", "🐾"].includes(top("犬の散歩")));
  assert.ok(["☕", "🍵"].includes(top("café con leche")));
  assert.equal(top("réserver un vol pour Tokyo"), "✈️");
});

test("ranking", () => {
  const results = model.suggestions("Pay my bills", 3);
  assert.equal(results.length, 3);
  assert.ok(results[0].confidence >= results[1].confidence);
  assert.ok(results.every((s) => s.confidence >= 0 && s.confidence <= 1));
});

test("empty input", () => {
  assert.deepEqual(model.suggestions("   "), []);
});

test("hub fetch + cache", { skip: process.env.HF_TOKEN ? false : "no HF_TOKEN" }, async () => {
  const remote = await load({ localModelPath: undefined });
  assert.equal(remote.suggestions("book a flight to Tokyo", 1)[0].emoji, "✈️");
});
