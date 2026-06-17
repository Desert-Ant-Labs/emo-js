import { webCache } from "./cache-web.js";
import { DEFAULT_HOST, DEFAULT_REPO, DEFAULT_REVISION, type EmoEnv, loadModel } from "./hub.js";
import { type EmoModel, type EmoSuggestion } from "./model.js";

export { createEmo, EmoModel, type EmoSuggestion, type EmoMeta } from "./model.js";
export { loadModel, type EmoEnv, type FileCache } from "./hub.js";

/** Loading configuration. Mutate before the first call, or pass overrides to {@link load}. */
export const env: EmoEnv = {
  host: DEFAULT_HOST,
  repo: DEFAULT_REPO,
  revision: DEFAULT_REVISION,
  allowRemote: true,
  useCache: true,
};

/** Loads the model from the Hugging Face Hub (cached in Cache Storage). */
export async function load(options: Partial<EmoEnv> = {}): Promise<EmoModel> {
  const e = { ...env, ...options };
  return loadModel(e, e.useCache ? webCache() : null);
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

export const Emo = { suggestions, load };
