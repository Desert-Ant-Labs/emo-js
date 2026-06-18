import { ngramEncode, SemTokenizer } from "./tokenizer.js";

/** A single emoji suggestion. */
export interface EmoSuggestion {
  /** The suggested emoji. */
  emoji: string;
  /** The model's normalized confidence, from `0` to `1`. */
  confidence: number;
}

export interface EmoMeta {
  labels: string[];
  n_hashes: number;
  n_buckets: number;
  n_importance: number;
  sem_dim: number;
  sem_pad_index: number;
}

// A weight tensor, either raw F32 or a 4-bit k-means palette kept packed in memory
// (a U8 index per weight, 2 per byte) and decoded on access — ~8x less memory than
// expanding to floats, for ~0.1ms more per inference.
class Tensor {
  constructor(
    readonly rows: number,
    readonly cols: number,
    private readonly floats: Float32Array | null,
    private readonly packed: Uint8Array | null,
    private readonly palette: Float32Array | null,
  ) {}
  get(i: number): number {
    const f = this.floats;
    if (f !== null) return f[i];
    const byte = this.packed![i >> 1];
    return this.palette![(i & 1) ? (byte >> 4) & 0xf : byte & 0xf];
  }
}

function erf(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}
const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2));

// Minimal safetensors reader: u64 header length, JSON header, then raw tensor bytes.
// Each weight is either raw F32, or a 4-bit k-means palette stored as a packed U8 index
// tensor plus a "<name>.palette" F32 tensor (logical 2-D shape kept in __metadata__).
// Palette weights stay packed and are decoded on access (see Tensor).
function parseWeights(bytes: Uint8Array) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerLen)));
  const dataStart = 8 + headerLen;
  const meta = header.__metadata__ ?? {};
  const slice = (name: string): { shape: number[]; bytes: Uint8Array } => {
    const e = header[name];
    if (!e) throw new Error(`emo: missing tensor ${name}`);
    const [a, b] = e.data_offsets;
    const out = new Uint8Array(b - a);
    out.set(bytes.subarray(dataStart + a, dataStart + b));
    return { shape: e.shape, bytes: out };
  };
  const f32 = (name: string): Float32Array => new Float32Array(slice(name).bytes.buffer);
  const expand = (name: string): Tensor => {
    if (header[name + ".palette"]) {
      const [rows, cols] = (meta["shape." + name] as string).split(",").map(Number);
      return new Tensor(rows, cols, null, slice(name).bytes, f32(name + ".palette"));
    }
    const [rows, cols = 1] = slice(name).shape;
    return new Tensor(rows, cols, f32(name), null, null);
  };
  return {
    embed: expand("embed"), sem: expand("sem"), importance: expand("importance"),
    w1: expand("w1"), b1: expand("b1"), w2: expand("w2"), b2: expand("b2"),
  };
}

/** Loads the bundled model and runs emoji inference. Construct once and reuse. */
export class EmoModel {
  private readonly tok: SemTokenizer;
  private readonly w;
  private readonly maxLen = 1024;

  constructor(weights: Uint8Array, tokenizer: Uint8Array, private readonly meta: EmoMeta) {
    this.w = parseWeights(weights);
    this.tok = new SemTokenizer(tokenizer);
  }

  /** Returns up to `limit` emoji suggestions, most likely first. */
  suggestions(text: string, limit = 3): EmoSuggestion[] {
    const trimmed = text.trim();
    if (!trimmed) return [];

    const { embed, sem, importance, w1, b1, w2, b2 } = this.w;
    const ngDim = embed.cols;
    const semDim = sem.cols;

    const ng = new Float32Array(ngDim);
    const f = ngramEncode(trimmed, this.meta.n_buckets, this.meta.n_hashes, this.meta.n_importance, this.maxLen);
    for (let i = 0; i < f.buckets.length; i++) {
      const im = f.importance[i];
      for (let k = 0; k < this.meta.n_hashes; k++) {
        const w = importance.get(im * importance.cols + k) * f.signs[i][k];
        const base = f.buckets[i][k] * ngDim;
        for (let c = 0; c < ngDim; c++) ng[c] += w * embed.get(base + c);
      }
    }
    const fcount = Math.max(1, f.buckets.length);
    for (let c = 0; c < ngDim; c++) ng[c] /= fcount;

    let ids = this.tok.encode(trimmed);
    if (ids.length > this.maxLen) ids = ids.slice(0, this.maxLen);
    if (ids.length === 0) ids = [this.meta.sem_pad_index];
    const sv = new Float32Array(semDim);
    for (const id of ids) {
      if (id >= sem.rows) continue;
      const base = id * semDim;
      for (let c = 0; c < semDim; c++) sv[c] += sem.get(base + c);
    }
    for (let c = 0; c < semDim; c++) sv[c] /= ids.length;
    let norm = 0;
    for (let c = 0; c < semDim; c++) norm += sv[c] * sv[c];
    norm = Math.sqrt(norm) + 1e-9;
    for (let c = 0; c < semDim; c++) sv[c] /= norm;

    const inDim = ngDim + semDim;
    const x = new Float32Array(inDim);
    x.set(ng, 0);
    x.set(sv, ngDim);

    const hid = w1.rows;
    const h = new Float32Array(hid);
    for (let o = 0; o < hid; o++) {
      let acc = b1.get(o);
      const base = o * inDim;
      for (let i = 0; i < inDim; i++) acc += w1.get(base + i) * x[i];
      h[o] = gelu(acc);
    }

    const n = w2.rows;
    const logits = new Float32Array(n);
    let maxLogit = -Infinity;
    for (let o = 0; o < n; o++) {
      let acc = b2.get(o);
      const base = o * hid;
      for (let i = 0; i < hid; i++) acc += w2.get(base + i) * h[i];
      logits[o] = acc;
      if (acc > maxLogit) maxLogit = acc;
    }
    let sum = 0;
    for (let o = 0; o < n; o++) { logits[o] = Math.exp(logits[o] - maxLogit); sum += logits[o]; }

    const labels = this.meta.labels;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => logits[b] - logits[a]);
    return order.slice(0, Math.max(0, limit)).map((i) => ({ emoji: labels[i], confidence: logits[i] / sum }));
  }
}

/** Creates an {@link EmoModel} from raw buffers (lowest-level entry). */
export function createEmo(buffers: { weights: Uint8Array; tokenizer: Uint8Array; meta: EmoMeta }): EmoModel {
  return new EmoModel(buffers.weights, buffers.tokenizer, buffers.meta);
}
