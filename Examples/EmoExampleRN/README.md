# Emo (React Native / Expo)

A tiny Expo app that predicts an emoji for whatever you type, on-device with
[`@desert-ant-labs/emo`](https://www.npmjs.com/package/@desert-ant-labs/emo).

This example uses the **fully-offline bundled-assets path**: the model
(`emo.safetensors`, `emo_tokenizer.bin`, `emo_meta.json`) ships inside the app and is
resolved through `env.readLocal`, so it **never touches the network** — no Hugging
Face Hub request, works in airplane mode on first launch. Inference is pure JS, so
there's no native module to link.

> Prefer downloading on first run and caching across launches instead of bundling?
> Drop the `env.readLocal`/asset wiring and set
> `env.cache = expoFileSystemCache(FileSystem)` — see the main README's React Native
> section.

## Run

From this folder:

```bash
# 1. install dependencies (expo install aligns versions to the Expo SDK)
npm install
npx expo install expo-asset expo-file-system

# 2. add the model files (see assets/README.md)
hf download desert-ant-labs/emo \
  emo.safetensors emo_tokenizer.bin emo_meta.json \
  --local-dir assets

# 3. start
npx expo start            # then press i (iOS), a (Android), or scan in Expo Go
```

## How it works

- `metro.config.js` — registers `.safetensors`/`.bin` as Metro asset extensions so
  `require()` returns a bundled asset.
- `emo.js` — sets `env.allowRemote = false` and `env.readLocal` to resolve each model
  file from the bundle: the binaries via `expo-asset` + `expo-file-system` (decoded
  with the package's `fromBase64`), and the JSON meta via a normal import.
- `App.js` — warms up the model once, then debounces `suggestions(text, 3)` while you
  type and shows the top emojis.

## Notes

- The model is bundled, so the app download is ~3.8 MB larger.
- On React Native older than ~0.74, add a `TextEncoder`/`TextDecoder` polyfill (e.g.
  `text-encoding`) — Hermes ships them on current versions.
