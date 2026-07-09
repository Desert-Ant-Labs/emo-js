import { initUsage, type UsageClient } from "@desert-ant-labs/desert-ant-web";

import { webCache } from "./cache-web.js";
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
};

let usage: UsageClient | null = null;

function instrument(model: EmoModel, usageKey?: string): EmoModel {
  const suggestions = model.suggestions.bind(model);
  model.suggestions = (text, limit, options) => {
    usage ??= initUsage({ key: usageKey });
    const out = suggestions(text, limit, options);
    usage.recordCall();
    return out;
  };
  return model;
}

/** Loads the model from the Hugging Face Hub (cached in Cache Storage). */
export async function load(options: Partial<EmoEnv> & { usageKey?: string } = {}): Promise<EmoModel> {
  const e = { ...env, ...options };
  return instrument(await loadModel(e, e.useCache ? webCache() : null), options.usageKey);
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
