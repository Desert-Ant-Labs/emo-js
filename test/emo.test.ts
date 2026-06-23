import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { before, test } from "node:test";
import { fileURLToPath } from "node:url";

import { applyEmojiSkinTone, load, type EmoModel } from "../src/index.node.js";

// Offline: load from local files so tests don't need network or a token.
const fixtures = fileURLToPath(new URL("../data", import.meta.url));
const hasFixtures = existsSync(join(fixtures, "emo.safetensors"));
const offline = hasFixtures
  ? {}
  : { skip: "no model fixtures in data/ — run `hf download desert-ant-labs/emo emo.safetensors emo_tokenizer.bin emo_meta.json --local-dir data`" };

let model: EmoModel;
before(async () => {
  if (hasFixtures) model = await load({ localModelPath: fixtures, allowRemote: false });
});

const top = (text: string) => model.suggestions(text, 1)[0]?.emoji ?? "";

test("english predictions", offline, () => {
  assert.ok(["💰", "💳", "🧾", "🏦"].includes(top("Pay my bills")));
  assert.ok(["🐕", "🐾", "🚶"].includes(top("walk the dog")));
  assert.equal(top("book a flight to Tokyo"), "✈️");
  assert.ok(["🦷", "📅", "🏥"].includes(top("dentist appointment")));
});

test("multilingual predictions", offline, () => {
  assert.ok(["🐕", "🐾"].includes(top("犬の散歩")));
  assert.ok(["☕", "🍵", "🥛"].includes(top("café con leche")));
  assert.equal(top("réserver un vol pour Tokyo"), "✈️");
});

test("ranking", offline, () => {
  const results = model.suggestions("Pay my bills", 3);
  assert.equal(results.length, 3);
  assert.ok(results[0].confidence >= results[1].confidence);
  assert.ok(results.every((s) => s.confidence >= 0 && s.confidence <= 1));
});

test("empty input", offline, () => {
  assert.deepEqual(model.suggestions("   "), []);
});

test("skin tone postprocessing", () => {
  assert.equal(applyEmojiSkinTone("🏃", "medium"), "🏃🏽");
  assert.equal(applyEmojiSkinTone("🧑‍🍳", "dark"), "🧑🏿‍🍳");
  assert.equal(applyEmojiSkinTone("✍️", "light"), "✍🏻");
  assert.equal(applyEmojiSkinTone("🐕", "medium"), "🐕");
});

test("hub fetch + cache", { skip: process.env.HF_TOKEN ? false : "no HF_TOKEN" }, async () => {
  const remote = await load({ localModelPath: undefined });
  assert.equal(remote.suggestions("book a flight to Tokyo", 1)[0].emoji, "✈️");
});
