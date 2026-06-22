import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { expoFileSystemCache, fromBase64, memoryCache, toBase64, type ExpoFileSystemLike } from "../src/cache-rn.js";
import { load as loadNode } from "../src/index.node.js";
import { load as loadRn } from "../src/index.rn.js";

const fixtures = fileURLToPath(new URL("../data", import.meta.url));
const FILES = ["emo.safetensors", "emo_tokenizer.bin", "emo_meta.json"] as const;
const hasFixtures = FILES.every((f) => existsSync(join(fixtures, f)));
const needsFixtures = hasFixtures
  ? {}
  : { skip: "no model fixtures in data/ — run `hf download desert-ant-labs/emo emo.safetensors emo_tokenizer.bin emo_meta.json --local-dir data`" };

test("base64 round-trips and matches Buffer", () => {
  for (const len of [0, 1, 2, 3, 4, 5, 255, 1000]) {
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = (i * 37 + 11) & 0xff;
    const b64 = toBase64(bytes);
    assert.equal(b64, Buffer.from(bytes).toString("base64"), `encode len=${len}`);
    assert.deepEqual(fromBase64(b64), bytes, `decode len=${len}`);
  }
});

test("memoryCache stores and returns bytes", async () => {
  const cache = memoryCache();
  assert.equal(await cache.get("u"), null);
  const data = new Uint8Array([1, 2, 3, 250]);
  await cache.put("u", data);
  assert.deepEqual(await cache.get("u"), data);
});

test("expoFileSystemCache round-trips via a fake expo-file-system", async () => {
  const files = new Map<string, string>();
  const fakeFs: ExpoFileSystemLike = {
    cacheDirectory: "file:///cache/",
    async makeDirectoryAsync() {},
    async readAsStringAsync(uri) {
      const v = files.get(uri);
      if (v == null) throw new Error("ENOENT");
      return v;
    },
    async writeAsStringAsync(uri, data) {
      files.set(uri, data);
    },
  };
  const cache = expoFileSystemCache(fakeFs);
  const url = "https://huggingface.co/desert-ant-labs/emo/resolve/main/emo_meta.json";
  assert.equal(await cache.get(url), null);
  const data = new Uint8Array([0, 1, 127, 128, 255, 42]);
  await cache.put(url, data);
  assert.deepEqual(await cache.get(url), data);
});

test("react-native load yields identical suggestions to node", needsFixtures, async () => {
  const bytesByName = new Map<string, Uint8Array>();
  for (const name of FILES) bytesByName.set(name, new Uint8Array(await readFile(join(fixtures, name))));

  const node = await loadNode({ localModelPath: fixtures, allowRemote: false });
  // RN has no filesystem: feed the same bytes through the readLocal hook instead.
  const rn = await loadRn({ allowRemote: false, readLocal: async (name) => bytesByName.get(name) ?? null });

  for (const phrase of ["walk the dog", "Pay my bills", "book a flight to Tokyo", "déneiger l'entrée", "犬の散歩"]) {
    assert.deepEqual(rn.suggestions(phrase, 3), node.suggestions(phrase, 3), `suggestions mismatch for "${phrase}"`);
  }
});
