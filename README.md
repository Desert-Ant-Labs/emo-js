# @desert-ant-labs/emo

On-device emoji suggestions from text. Suggests the best-matching emoji for short
tasks, calendar entries, notes, or message drafts across **23 languages** — fully
in-process, no inference runtime.

```ts
import { suggestions } from "@desert-ant-labs/emo";

const results = await suggestions("Pay my bills");
// [{ emoji: "💰", confidence: 0.62 }, ...]

const emoji = (await suggestions("犬の散歩", 1))[0]?.emoji; // "🐕"
const toned = (await suggestions("go for a run", 1, { skinTone: "medium" }))[0]?.emoji; // "🏃🏽"
```

## Features

- Pure-JS inference (no ONNX/WASM runtime) — prediction is sub-millisecond
- Suggests from a data-driven vocabulary of ~300 task/calendar/message emojis
- Supports 23 languages (incl. CJK, Arabic, Thai, Hindi, …)
- Model (~5.0 MB, 4-bit palettized) is fetched from the Hugging Face Hub at a **pinned revision**, then
  cached — to the **filesystem** on Node and to **Cache Storage** in the browser, so
  it loads once and runs offline after

## Install

```bash
npm install @desert-ant-labs/emo
```

## Importing

Pure ESM and fully tree-shakeable. The **same import works everywhere** — Node, bundlers,
browsers, and edge/worker runtimes — the right build is selected automatically:

```ts
import { suggestions } from "@desert-ant-labs/emo";
```

CommonJS consumers use dynamic import (`const { suggestions } = await import("@desert-ant-labs/emo")`);
native `require()` works on Node ≥ 22.12.

**Minimal footprint / bring-your-own-bytes.** If you load the model files yourself, import the
hub-free core from `@desert-ant-labs/emo/core` — only the ~6 KB inference engine, with zero
network/filesystem code:

```ts
import { createEmo } from "@desert-ant-labs/emo/core";

// weights and tokenizer are Uint8Array buffers; meta is the parsed emo_meta.json
const emo = createEmo({ weights, tokenizer, meta });
emo.suggestions("walk the dog", 1)[0]?.emoji; // "🐕"
```

## Loading model

Model files are fetched from the Hugging Face Hub
([`desert-ant-labs/emo`](https://huggingface.co/desert-ant-labs/emo)) at a pinned
revision and cached. The first call downloads `emo.safetensors`, `emo_tokenizer.bin`,
and `emo_meta.json`; subsequent calls read from the cache.

- **Node**: `suggestions()` works zero-config; files cache under `~/.cache/emo`.
  To run fully offline, ship the files yourself and point at a folder with
  `env.localModelPath` (or `EMO_LOCAL_PATH`).
- **Browser**: same API; files cache in Cache Storage.

```ts
import { env, load, suggestions } from "@desert-ant-labs/emo";

// optional global config (set before first use)
env.revision = "main";              // or a commit SHA / tag
env.cacheDir = "/var/cache/emo";    // Node only
env.localModelPath = "./emo-model"; // Node: use local files, skip the Hugging Face Hub

// or load an explicit instance (synchronous inference after it resolves)
const emo = await load({ revision: "main" });
emo.suggestions("book a flight to Tokyo", 1)[0]?.emoji; // "✈️"
```

## API

```ts
export function suggestions(text: string, limit?: number, options?: EmoSuggestionOptions): Promise<EmoSuggestion[]>;
export function load(options?: Partial<EmoEnv>): Promise<EmoModel>;
export function createEmo(buffers: { weights; tokenizer; meta }): EmoModel; // raw buffers
export const env: EmoEnv;
export function reset(): void; // clear the memoized model so the next suggestions() re-reads env

export interface EmoSuggestion {
  emoji: string;
  confidence: number;
}

export interface EmoSuggestionOptions {
  skinTone?: EmojiSkinTone; // default: "default"
}

export type EmojiSkinTone = "default" | "light" | "mediumLight" | "medium" | "mediumDark" | "dark";
```

`suggestions(text, limit = 3, options)` returns up to `limit` emojis, most likely first;
empty input returns `[]`. `skinTone` post-processes skin-tone-capable emoji; the default is `"default"` (no modifier). `EmoModel.suggestions` is synchronous once loaded.

## Example

[`examples/EmoExample`](examples/EmoExample) is a small todo-list web app that predicts an
emoji for each task on-device. Run it with `node server.js` from that folder and open
<http://localhost:5173>.

## Model

Published at [`desert-ant-labs/emo`](https://huggingface.co/desert-ant-labs/emo) on Hugging Face.

## License

See [`LICENSE.md`](LICENSE.md) — Desert Ant Labs Source-Available License v1.0. Free for commercial use up to 100,000 MAU per Model; <licensing@desertant.ai> above that.
