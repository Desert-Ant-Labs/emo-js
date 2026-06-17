import { createEmo, EmoModel, type EmoMeta } from "./model.js";

export const DEFAULT_HOST = "https://huggingface.co";
export const DEFAULT_REPO = "desert-ant-labs/emo";
/** Pinned revision of the model repo (a commit SHA). */
export const DEFAULT_REVISION = "755daf6a6a3a6737024cd7bf1a26730f9813a26e";

const FILES = ["emo_weights.bin", "emo_tokenizer.bin", "emo_meta.json"] as const;

/** Resolution + caching configuration (mutate the exported `env` to change defaults). */
export interface EmoEnv {
  /** Hugging Face host serving the model repo. */
  host: string;
  /** Model repo id, e.g. `"desert-ant-labs/emo"`. */
  repo: string;
  /** Pinned revision (commit SHA, tag, or branch). */
  revision: string;
  /** Allow fetching from the Hugging Face Hub. Set `false` to require a local copy. */
  allowRemote: boolean;
  /** Cache downloaded files (filesystem on Node, Cache Storage in the browser). */
  useCache: boolean;
  /** Directory of pre-downloaded model files to use instead of the Hugging Face Hub (Node). */
  localModelPath?: string;
  /** Filesystem cache directory (Node). */
  cacheDir?: string;
  /** Optional Hugging Face access token (Node). */
  token?: string;
}

/** A key/value store for cached file bytes, keyed by resolve URL. */
export interface FileCache {
  get(key: string): Promise<Uint8Array | null>;
  put(key: string, data: Uint8Array): Promise<void>;
}

function resolveUrl(env: EmoEnv, name: string): string {
  return `${env.host}/${env.repo}/resolve/${env.revision}/${name}`;
}

async function fetchFile(
  env: EmoEnv,
  name: string,
  cache: FileCache | null,
  readLocal?: (name: string) => Promise<Uint8Array | null>,
): Promise<Uint8Array> {
  if (readLocal) {
    const local = await readLocal(name);
    if (local) return local;
  }
  const url = resolveUrl(env, name);
  if (cache && env.useCache) {
    const hit = await cache.get(url);
    if (hit) return hit;
  }
  if (!env.allowRemote) throw new Error(`emo: ${name} unavailable locally and remote loading is disabled`);
  const res = await fetch(url, env.token ? { headers: { Authorization: `Bearer ${env.token}` } } : undefined);
  if (!res.ok) throw new Error(`emo: failed to fetch ${name} from ${url} (${res.status} ${res.statusText})`);
  const data = new Uint8Array(await res.arrayBuffer());
  if (cache && env.useCache) {
    try {
      await cache.put(url, data);
    } catch {
      /* caching is best-effort */
    }
  }
  return data;
}

/** Resolves all model files (local dir → cache → Hugging Face Hub) and builds an {@link EmoModel}. */
export async function loadModel(
  env: EmoEnv,
  cache: FileCache | null,
  readLocal?: (name: string) => Promise<Uint8Array | null>,
): Promise<EmoModel> {
  const [weights, tokenizer, meta] = await Promise.all(
    FILES.map((name) => fetchFile(env, name, cache, readLocal)),
  );
  return createEmo({ weights, tokenizer, meta: JSON.parse(new TextDecoder().decode(meta)) as EmoMeta });
}
