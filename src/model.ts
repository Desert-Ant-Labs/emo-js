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

interface F32Tensor { data: Float32Array; rows: number; cols: number }
interface Int8Tensor { data: Int8Array; scale: Float32Array; rows: number; cols: number }

function erf(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}
const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2));

function parseWeights(bytes: Uint8Array) {
  if (bytes[0] !== 0x45 || bytes[1] !== 0x4d || bytes[2] !== 0x57 || bytes[3] !== 0x31) {
    throw new Error("bad weights file");
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = dv.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerLen)));
  const dataStart = 8 + headerLen;
  const copy = (start: number, len: number): ArrayBuffer => {
    const out = new Uint8Array(len);
    out.set(bytes.subarray(start, start + len));
    return out.buffer;
  };
  const f32 = (name: string): F32Tensor => {
    const h = header[name];
    const [rows, cols = 1] = h.shape;
    return { data: new Float32Array(copy(dataStart + h.off, rows * cols * 4)), rows, cols };
  };
  const int8 = (name: string): Int8Tensor => {
    const h = header[name];
    const [rows, cols] = h.shape;
    return {
      data: new Int8Array(copy(dataStart + h.off, rows * cols)),
      scale: new Float32Array(copy(dataStart + h.scaleOff, rows * 4)),
      rows, cols,
    };
  };
  return {
    embed: int8("embed"), sem: int8("sem"), importance: f32("importance"),
    w1: f32("w1"), b1: f32("b1"), w2: f32("w2"), b2: f32("b2"),
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
        const w = importance.data[im * importance.cols + k] * f.signs[i][k];
        const b = f.buckets[i][k];
        const scale = embed.scale[b];
        const base = b * ngDim;
        for (let c = 0; c < ngDim; c++) ng[c] += w * (embed.data[base + c] * scale);
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
      const scale = sem.scale[id];
      const base = id * semDim;
      for (let c = 0; c < semDim; c++) sv[c] += sem.data[base + c] * scale;
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
      let acc = b1.data[o];
      const base = o * inDim;
      for (let i = 0; i < inDim; i++) acc += w1.data[base + i] * x[i];
      h[o] = gelu(acc);
    }

    const n = w2.rows;
    const logits = new Float32Array(n);
    let maxLogit = -Infinity;
    for (let o = 0; o < n; o++) {
      let acc = b2.data[o];
      const base = o * hid;
      for (let i = 0; i < hid; i++) acc += w2.data[base + i] * h[i];
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
