import { homedir } from "node:os";
import { join } from "node:path";

import { fsCache, localReader } from "./cache-node.js";
import { DEFAULT_HOST, DEFAULT_REPO, DEFAULT_REVISION, type EmoEnv, loadModel } from "./hub.js";
import { type EmoModel, type EmoSuggestion } from "./model.js";
import { type EmoSuggestionOptions } from "./skin-tone.js";

export { createEmo, EmoModel, type EmoSuggestion, type EmoMeta } from "./model.js";
export { applyEmojiSkinTone, type EmojiSkinTone, type EmoSuggestionOptions } from "./skin-tone.js";
export { loadModel, type EmoEnv, type FileCache } from "./hub.js";

/** Loading configuration. Mutate before the first call, or pass overrides to {@link load}. */
export const env: EmoEnv = {
  host: DEFAULT_HOST,
  repo: DEFAULT_REPO,
  revision: DEFAULT_REVISION,
  allowRemote: true,
  useCache: true,
  cacheDir: process.env.EMO_CACHE_DIR ?? join(homedir(), ".cache", "emo"),
  localModelPath: process.env.EMO_LOCAL_PATH,
  token: process.env.HF_TOKEN ?? process.env.HUGGING_FACE_HUB_TOKEN,
};

/** Loads the model: a local dir if configured, else the Hugging Face Hub (cached to disk). */
export async function load(options: Partial<EmoEnv> = {}): Promise<EmoModel> {
  const e = { ...env, ...options };
  const cache = e.useCache && e.cacheDir ? fsCache(e.cacheDir) : null;
  const readLocal = e.localModelPath ? localReader(e.localModelPath) : undefined;
  return loadModel(e, cache, readLocal);
}

let modelPromise: Promise<EmoModel> | null = null;

/**
 * Returns emoji suggestions for a phrase, most likely first. The model is loaded
 * (and cached) lazily on first call. Empty input returns `[]`.
 */
export async function suggestions(text: string, limit = 3, options: EmoSuggestionOptions = {}): Promise<EmoSuggestion[]> {
  if (!modelPromise) {
    // Don't cache a rejected load: clear the slot on failure so the next call retries.
    modelPromise = load().catch((err) => {
      modelPromise = null;
      throw err;
    });
  }
  return (await modelPromise).suggestions(text, limit, options);
}

/** Clears the memoized model so the next {@link suggestions} call re-reads `env`. */
export function reset(): void {
  modelPromise = null;
}
