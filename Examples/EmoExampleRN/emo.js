// Wires @desert-ant-labs/emo to load its model from the app bundle — no network,
// fully offline. The three model files are shipped as bundled assets and resolved
// through `env.readLocal`, so nothing is ever fetched from the Hugging Face Hub.
import { Asset } from "expo-asset";
import * as FileSystem from "expo-file-system";
import { env, fromBase64, suggestions } from "@desert-ant-labs/emo";

import metaJson from "./assets/emo_meta.json";

// The binary weights/tokenizer are bundled as assets (see metro.config.js, which
// registers the `.safetensors` and `.bin` extensions so `require()` returns an asset).
const binaryAssets = {
  "emo.safetensors": require("./assets/emo.safetensors"),
  "emo_tokenizer.bin": require("./assets/emo_tokenizer.bin"),
};

async function readAssetBytes(mod) {
  const asset = Asset.fromModule(mod);
  await asset.downloadAsync(); // copies the bundled asset to a readable localUri
  const b64 = await FileSystem.readAsStringAsync(asset.localUri ?? asset.uri, {
    encoding: FileSystem.EncodingType.Base64,
  });
  return fromBase64(b64);
}

// Resolve every model file locally; never touch the network.
env.allowRemote = false;
env.readLocal = async (name) => {
  // Metro parses `.json` as a JS value, so re-encode the meta to bytes (the loader
  // parses it back). The binaries come straight from the bundle.
  if (name === "emo_meta.json") return new TextEncoder().encode(JSON.stringify(metaJson));
  const mod = binaryAssets[name];
  return mod ? readAssetBytes(mod) : null;
};

export { suggestions };
