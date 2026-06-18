const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const MASK = (1n << 64n) - 1n;

function fnv64(s: string, seed: bigint): bigint {
  let h = (FNV_OFFSET ^ seed) & MASK;
  for (const b of new TextEncoder().encode(s)) {
    h = ((h ^ BigInt(b)) * FNV_PRIME) & MASK;
  }
  return h;
}

const BUCKET_SEEDS = [
  0x9e3779b97f4a7c15n, 0xc2b2ae3d27d4eb4fn, 0x165667b19e3779f9n,
  0x27d4eb2f165667c5n, 0x85ebca77c2b2ae63n,
];
const IMP_SEED = 0xff51afd7ed558ccdn;

const NA: [number, number] = [3, 5];
const NC: [number, number] = [1, 2];
const NJ: [number, number] = [2, 4];
const NS: [number, number] = [2, 4];
const NI: [number, number] = [1, 3];

const reWord = /\p{L}|\p{M}|\p{N}/u;
const reMark = /\p{M}/u;

const cp = (s: string) => s.codePointAt(0)!;
const isHangul = (s: string) => cp(s) >= 0xac00 && cp(s) <= 0xd7a3;
function isCJK(s: string): boolean {
  const v = cp(s);
  return (v >= 0x4e00 && v <= 0x9fff) || (v >= 0x3400 && v <= 0x4dbf)
    || (v >= 0x20000 && v <= 0x2a6df) || (v >= 0xf900 && v <= 0xfaff)
    || (v >= 0x3040 && v <= 0x30ff) || (v >= 0x31f0 && v <= 0x31ff)
    || isHangul(s);
}
function isSEA(s: string): boolean {
  const v = cp(s);
  return (v >= 0x0e00 && v <= 0x0eff) || (v >= 0x1000 && v <= 0x109f) || (v >= 0x1780 && v <= 0x17ff);
}
const isIndic = (s: string) => cp(s) >= 0x0900 && cp(s) <= 0x0dff;

function jamo(c: string): string[] {
  if (!isHangul(c)) return [c];
  const s = cp(c) - 0xac00;
  const r = [String.fromCodePoint(0x1100 + Math.floor(s / 588)), String.fromCodePoint(0x1161 + Math.floor((s % 588) / 28))];
  if (s % 28 !== 0) r.push(String.fromCodePoint(0x11a7 + (s % 28)));
  return r;
}

function normalize(text: string): string {
  return text.normalize("NFKC").toLowerCase().split(/\s+/).filter(Boolean).join(" ");
}

function tokens(text: string): string[][] {
  const out: string[][] = [];
  let cur: string[] = [];
  for (const c of text) {
    if (reWord.test(c)) cur.push(c);
    else if (cur.length) { out.push(cur); cur = []; }
  }
  if (cur.length) out.push(cur);
  return out;
}

function clusters(s: string[]): string[][] {
  const out: string[][] = [];
  let cur: string[] = [];
  for (const c of s) {
    if (!cur.length) { cur = [c]; continue; }
    const p = cp(cur[cur.length - 1]);
    const vir = p >= 0x0900 && p <= 0x0dff && ((p & 0xff) === 0x4d || (p & 0xff) === 0xcd);
    if (reMark.test(c) || vir) cur.push(c);
    else { out.push(cur); cur = [c]; }
  }
  if (cur.length) out.push(cur);
  return out;
}

function charGrams(s: string[], lo: number, hi: number, tag: string): string[] {
  const r: string[] = [];
  for (let n = lo; n <= hi; n++) {
    for (let i = 0; i + n <= s.length; i++) r.push(tag + s.slice(i, i + n).join(""));
  }
  return r;
}

function clusterGrams(cl: string[][], lo: number, hi: number, tag: string): string[] {
  const r: string[] = [];
  for (let n = lo; n <= hi; n++) {
    for (let i = 0; i + n <= cl.length; i++) r.push(tag + cl.slice(i, i + n).flat().join(""));
  }
  return r;
}

function feats(text: string): string[] {
  const out: string[] = [];
  for (const run of tokens(normalize(text))) {
    if (run.some(isSEA)) {
      out.push(...charGrams(run, NS[0], NS[1], "s:"));
    } else if (run.some(isIndic)) {
      const cl = clusters(run);
      out.push("a:" + run.join(""));
      out.push(...clusterGrams(cl, 1, 1, "k:"));
      out.push(...clusterGrams([["<"], ...cl, [">"]], 2, NI[1], "k:"));
    } else if (run.some(isCJK)) {
      const ex: string[] = [];
      for (const c of run) {
        if (isHangul(c)) out.push(...charGrams(jamo(c), NJ[0], NJ[1], "j:"));
        ex.push(c);
      }
      out.push(...charGrams(ex, NC[0], NC[1], "c:"));
    } else {
      out.push("w:" + run.join(""));
      out.push(...charGrams(["<", ...run, ">"], NA[0], NA[1], "g:"));
    }
  }
  return out.length ? out : ["w:\u0000"];
}

export interface NGramFeatures {
  buckets: Int32Array[];
  signs: Float32Array[];
  importance: Int32Array;
}

export function ngramEncode(
  text: string, nBuckets: number, nHashes: number, nImportance: number, maxFeatures: number,
): NGramFeatures {
  let fs = feats(text);
  if (fs.length > maxFeatures) fs = fs.slice(0, maxFeatures);
  const buckets: Int32Array[] = [];
  const signs: Float32Array[] = [];
  const importance = new Int32Array(fs.length);
  const nb = BigInt(nBuckets);
  const ni = BigInt(nImportance);
  for (let f = 0; f < fs.length; f++) {
    const bk = new Int32Array(nHashes);
    const sg = new Float32Array(nHashes);
    for (let k = 0; k < nHashes; k++) {
      const h = fnv64(fs[f], BUCKET_SEEDS[k]);
      bk[k] = Number(h % nb);
      sg[k] = (h >> 63n) & 1n ? 1 : -1;
    }
    buckets.push(bk);
    signs.push(sg);
    importance[f] = Number(fnv64(fs[f], IMP_SEED) % ni);
  }
  return { buckets, signs, importance };
}

export class SemTokenizer {
  private pieces: string[] = [];
  private scores: Float32Array;
  private index = new Map<string, number>();
  private unkId: number;
  private unkScore: number;
  private maxLen = 1;

  constructor(data: Uint8Array) {
    const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
    if (data[0] !== 0x45 || data[1] !== 0x4d || data[2] !== 0x54 || data[3] !== 0x4b) {
      throw new Error("bad tokenizer file");
    }
    this.unkId = dv.getInt32(6, true);
    const k = dv.getUint32(10, true);
    let off = 14;
    this.scores = new Float32Array(k);
    for (let i = 0; i < k; i++) { this.scores[i] = dv.getFloat32(off, true); off += 4; }
    const lens = new Array<number>(k);
    for (let i = 0; i < k; i++) { lens[i] = dv.getUint16(off, true); off += 2; }
    const dec = new TextDecoder();
    for (let i = 0; i < k; i++) {
      const piece = dec.decode(data.subarray(off, off + lens[i]));
      off += lens[i];
      this.pieces.push(piece);
      this.index.set(piece, i);
      const n = [...piece].length;
      if (n > this.maxLen) this.maxLen = n;
    }
    this.maxLen = Math.min(this.maxLen, 24);
    this.unkScore = this.scores[this.unkId];
  }

  encode(text: string): number[] {
    const norm = "\u2581" + text.toLowerCase().normalize("NFKC").replace(/ /g, "\u2581");
    const s = [...norm];
    const n = s.length;
    if (n === 0) return [];
    const NEG = -1e18;
    const best = new Float64Array(n + 1).fill(NEG);
    best[0] = 0;
    const backPos = new Int32Array(n + 1).fill(-1);
    const backId = new Int32Array(n + 1).fill(-1);
    for (let i = 1; i <= n; i++) {
      const lo = Math.max(0, i - this.maxLen);
      for (let j = lo; j < i; j++) {
        const tid = this.index.get(s.slice(j, i).join(""));
        if (tid !== undefined) {
          const sc = best[j] + this.scores[tid];
          if (sc > best[i]) { best[i] = sc; backPos[i] = j; backId[i] = tid; }
        }
      }
      const cand = best[i - 1] + this.unkScore;
      if (cand > best[i]) { best[i] = cand; backPos[i] = i - 1; backId[i] = this.unkId; }
    }
    const ids: number[] = [];
    let i = n;
    while (i > 0) { ids.push(backId[i]); i = backPos[i]; }
    return ids.reverse();
  }
}
