import { memoryCache } from "./cache-rn.js";
import { DEFAULT_HOST, DEFAULT_REPO, DEFAULT_REVISION, type EmoEnv, type FileCache, loadModel } from "./hub.js";
import { type EmoModel, type EmoSuggestion } from "./model.js";

export { createEmo, EmoModel, type EmoSuggestion, type EmoMeta } from "./model.js";
export { loadModel, type EmoEnv, type FileCache } from "./hub.js";
export { memoryCache, expoFileSystemCache, toBase64, fromBase64, type ExpoFileSystemLike } from "./cache-rn.js";

/** Loading configuration for React Native (extends {@link EmoEnv}). */
export interface EmoRnEnv extends EmoEnv {
  /**
   * Persistent cache for downloaded files. Defaults to an in-memory cache (per JS
   * session); set {@link expoFileSystemCache} to persist across launches.
   */
  cache?: FileCache;
  /**
   * Resolve a model file (`emo.safetensors`, `emo_tokenizer.bin`, `emo_meta.json`)
   * from bundled assets or local storage before the cache or the Hugging Face Hub
   * is consulted. Return `null` to fall through.
   */
  readLocal?: (name: string) => Promise<Uint8Array | null>;
}

/** Loading configuration. Mutate before the first call, or pass overrides to {@link load}. */
export const env: EmoRnEnv = {
  host: DEFAULT_HOST,
  repo: DEFAULT_REPO,
  revision: DEFAULT_REVISION,
  allowRemote: true,
  useCache: true,
  cache: memoryCache(),
};

/** Loads the model: bundled/local files if provided, else the Hugging Face Hub (cached via `env.cache`). */
export async function load(options: Partial<EmoRnEnv> = {}): Promise<EmoModel> {
  const e = { ...env, ...options };
  const cache = e.useCache ? (e.cache ?? null) : null;
  return loadModel(e, cache, e.readLocal);
}

let modelPromise: Promise<EmoModel> | null = null;

/**
 * Returns emoji suggestions for a phrase, most likely first. The model is loaded
 * (and cached) lazily on first call. Empty input returns `[]`.
 */
export async function suggestions(text: string, limit = 3): Promise<EmoSuggestion[]> {
  if (!modelPromise) {
    // Don't cache a rejected load: clear the slot on failure so the next call retries.
    modelPromise = load().catch((err) => {
      modelPromise = null;
      throw err;
    });
  }
  return (await modelPromise).suggestions(text, limit);
}

/** Clears the memoized model so the next {@link suggestions} call re-reads `env`. */
export function reset(): void {
  modelPromise = null;
}
