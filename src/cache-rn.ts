import type { FileCache } from "./hub.js";

/**
 * In-memory cache (the React Native default): downloaded files persist for the JS
 * session and are re-fetched after a cold start. Swap in {@link expoFileSystemCache}
 * (or any {@link FileCache}) via `env.cache` to persist across launches.
 */
export function memoryCache(): FileCache {
  const store = new Map<string, Uint8Array>();
  return {
    async get(key) {
      return store.get(key) ?? null;
    },
    async put(key, data) {
      store.set(key, data);
    },
  };
}

// Base64 codec: React Native lacks Buffer and has no reliable global atob/btoa for
// binary, and expo-file-system reads/writes file bytes as base64 strings.
const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64_REV = /* @__PURE__ */ (() => {
  const t = new Int16Array(128).fill(-1);
  for (let i = 0; i < B64.length; i++) t[B64.charCodeAt(i)] = i;
  return t;
})();

/** Encodes bytes to a standard (padded) base64 string. */
export function toBase64(bytes: Uint8Array): string {
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i];
    const b1 = i + 1 < bytes.length ? bytes[i + 1] : 0;
    const b2 = i + 2 < bytes.length ? bytes[i + 2] : 0;
    out += B64[b0 >> 2];
    out += B64[((b0 & 3) << 4) | (b1 >> 4)];
    out += i + 1 < bytes.length ? B64[((b1 & 15) << 2) | (b2 >> 6)] : "=";
    out += i + 2 < bytes.length ? B64[b2 & 63] : "=";
  }
  return out;
}

/** Decodes a base64 string (padding optional, whitespace ignored) to bytes. */
export function fromBase64(str: string): Uint8Array {
  const clean = str.replace(/[^A-Za-z0-9+/]/g, "");
  const out = new Uint8Array(Math.floor((clean.length * 3) / 4));
  let p = 0;
  for (let i = 0; i < clean.length; i += 4) {
    const c0 = B64_REV[clean.charCodeAt(i)];
    const c1 = B64_REV[clean.charCodeAt(i + 1)];
    const c2 = B64_REV[clean.charCodeAt(i + 2)];
    const c3 = B64_REV[clean.charCodeAt(i + 3)];
    out[p++] = (c0 << 2) | (c1 >> 4);
    if (p < out.length) out[p++] = ((c1 & 15) << 4) | (c2 >> 2);
    if (p < out.length) out[p++] = ((c2 & 3) << 6) | c3;
  }
  return out;
}

/** The slice of `expo-file-system` that {@link expoFileSystemCache} uses. */
export interface ExpoFileSystemLike {
  cacheDirectory: string | null;
  readAsStringAsync(uri: string, options: { encoding: "base64" }): Promise<string>;
  writeAsStringAsync(uri: string, data: string, options: { encoding: "base64" }): Promise<void>;
  makeDirectoryAsync(uri: string, options: { intermediates: boolean }): Promise<void>;
}

/**
 * Filesystem-backed cache for React Native, persisting downloads across launches.
 * Pass in the host app's `expo-file-system` module so this package stays
 * dependency-free:
 *
 * ```ts
 * import * as FileSystem from "expo-file-system";
 * import { env, expoFileSystemCache } from "@desert-ant-labs/emo";
 * env.cache = expoFileSystemCache(FileSystem);
 * ```
 */
export function expoFileSystemCache(
  FileSystem: ExpoFileSystemLike,
  dir = `${FileSystem.cacheDirectory ?? ""}emo/`,
): FileCache {
  const ready = FileSystem.makeDirectoryAsync(dir, { intermediates: true }).catch(() => {});
  const fileFor = (key: string) => dir + key.replace(/^https?:\/\//, "").replace(/[^a-zA-Z0-9._-]/g, "_");
  return {
    async get(key) {
      try {
        return fromBase64(await FileSystem.readAsStringAsync(fileFor(key), { encoding: "base64" }));
      } catch {
        return null;
      }
    },
    async put(key, data) {
      try {
        await ready;
        await FileSystem.writeAsStringAsync(fileFor(key), toBase64(data), { encoding: "base64" });
      } catch {
        /* caching is best-effort */
      }
    },
  };
}
