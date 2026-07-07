import { applyEmojiSkinTone, type EmoSuggestionOptions } from "./skin-tone.js";
import { ngramEncode, SemTokenizer } from "./tokenizer.js";

export { applyEmojiSkinTone, type EmojiSkinTone, type EmoSuggestionOptions } from "./skin-tone.js";

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
  /** "transformer" runs the semantic sequence through an encoder; absent = mean-pool. */
  arch?: string;
  n_layers?: number;
  n_heads?: number;
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

// ---- transformer helpers (semantic sequence path) -------------------------
// S is a row-major T x inDim buffer; returns T x outDim = S @ W^T + b (W is outDim x inDim).
function linear(S: Float32Array, T: number, inDim: number, W: Tensor, outDim: number, b: Tensor | null): Float32Array {
  const out = new Float32Array(T * outDim);
  for (let t = 0; t < T; t++) {
    const sb = t * inDim, ob = t * outDim;
    for (let o = 0; o < outDim; o++) {
      let acc = b ? b.get(o) : 0;
      const wb = o * inDim;
      for (let i = 0; i < inDim; i++) acc += W.get(wb + i) * S[sb + i];
      out[ob + o] = acc;
    }
  }
  return out;
}

// in-place post-norm LayerNorm over the last dim.
function layerNorm(S: Float32Array, T: number, d: number, w: Tensor, b: Tensor): void {
  for (let t = 0; t < T; t++) {
    const base = t * d;
    let mean = 0;
    for (let c = 0; c < d; c++) mean += S[base + c];
    mean /= d;
    let v = 0;
    for (let c = 0; c < d; c++) { const z = S[base + c] - mean; v += z * z; }
    const inv = 1 / Math.sqrt(v / d + 1e-5);
    for (let c = 0; c < d; c++) S[base + c] = (S[base + c] - mean) * inv * w.get(c) + b.get(c);
  }
}

interface TrLayer {
  qkv_w: Tensor; qkv_b: Tensor; o_w: Tensor; o_b: Tensor;
  f1_w: Tensor; f1_b: Tensor; f2_w: Tensor; f2_b: Tensor;
  n1_w: Tensor; n1_b: Tensor; n2_w: Tensor; n2_b: Tensor;
}

// Post-norm encoder layer (gelu FFN), matching nn.TransformerEncoderLayer.
function encoderLayer(S: Float32Array, T: number, d: number, H: number, L: TrLayer): Float32Array {
  const dh = d / H, scale = 1 / Math.sqrt(dh);
  const qkv = linear(S, T, d, L.qkv_w, 3 * d, L.qkv_b);      // T x 3d
  const attn = new Float32Array(T * d);
  const scores = new Float32Array(T);
  for (let h = 0; h < H; h++) {
    const off = h * dh;
    for (let ti = 0; ti < T; ti++) {
      const qb = ti * 3 * d + off;
      let mx = -Infinity;
      for (let tj = 0; tj < T; tj++) {
        const kb = tj * 3 * d + d + off;
        let s = 0;
        for (let c = 0; c < dh; c++) s += qkv[qb + c] * qkv[kb + c];
        s *= scale; scores[tj] = s; if (s > mx) mx = s;
      }
      let sum = 0;
      for (let tj = 0; tj < T; tj++) { const e = Math.exp(scores[tj] - mx); scores[tj] = e; sum += e; }
      const ob = ti * d + off;
      for (let tj = 0; tj < T; tj++) {
        const a = scores[tj] / sum, vb = tj * 3 * d + 2 * d + off;
        for (let c = 0; c < dh; c++) attn[ob + c] += a * qkv[vb + c];
      }
    }
  }
  const o = linear(attn, T, d, L.o_w, d, L.o_b);             // T x d
  for (let i = 0; i < T * d; i++) o[i] += S[i];
  layerNorm(o, T, d, L.n1_w, L.n1_b);
  const ff1 = linear(o, T, d, L.f1_w, L.f1_b.rows, L.f1_b);  // T x ff
  for (let i = 0; i < ff1.length; i++) ff1[i] = gelu(ff1[i]);
  const ff2 = linear(ff1, T, L.f1_b.rows, L.f2_w, d, L.f2_b);// T x d
  for (let i = 0; i < T * d; i++) ff2[i] += o[i];
  layerNorm(ff2, T, d, L.n2_w, L.n2_b);
  return ff2;
}

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
  const base = {
    embed: expand("embed"), sem: expand("sem"), importance: expand("importance"),
    w1: expand("w1"), b1: expand("b1"), w2: expand("w2"), b2: expand("b2"),
  };
  if (!header["ap_w"]) return { ...base, tr: null };
  // transformer encoder + attention pool
  const nLayers = Object.keys(header).filter((k) => /^L\d+_qkv_w$/.test(k)).length;
  const layers: TrLayer[] = [];
  for (let l = 0; l < nLayers; l++) {
    layers.push({
      qkv_w: expand(`L${l}_qkv_w`), qkv_b: expand(`L${l}_qkv_b`),
      o_w: expand(`L${l}_o_w`), o_b: expand(`L${l}_o_b`),
      f1_w: expand(`L${l}_f1_w`), f1_b: expand(`L${l}_f1_b`),
      f2_w: expand(`L${l}_f2_w`), f2_b: expand(`L${l}_f2_b`),
      n1_w: expand(`L${l}_n1_w`), n1_b: expand(`L${l}_n1_b`),
      n2_w: expand(`L${l}_n2_w`), n2_b: expand(`L${l}_n2_b`),
    });
  }
  const tr = { layers, ap_w: expand("ap_w"), ap_b: expand("ap_b"), ap_q: expand("ap_q"), pn_w: expand("pn_w"), pn_b: expand("pn_b") };
  return { ...base, tr };
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
  suggestions(text: string, limit = 3, options: EmoSuggestionOptions = {}): EmoSuggestion[] {
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
    const tr = this.w.tr;
    if (tr) {
      // semantic sequence -> transformer encoder -> attention pool -> LayerNorm
      const T = ids.length;
      let S: Float32Array<ArrayBufferLike> = new Float32Array(T * semDim);
      for (let t = 0; t < T; t++) {
        const id = ids[t] < sem.rows ? ids[t] : this.meta.sem_pad_index;
        const sb = id * semDim, ob = t * semDim;
        for (let c = 0; c < semDim; c++) S[ob + c] = id < sem.rows ? sem.get(sb + c) : 0;
      }
      const H = this.meta.n_heads ?? 4;
      for (const L of tr.layers) S = encoderLayer(S, T, semDim, H, L);
      // attention pool: score_t = (aproj(tanh(S_t)) . q) / sqrt(d), softmax over t, weighted sum
      const scoreScale = 1 / Math.sqrt(semDim);
      const tanhS = new Float32Array(T * semDim);
      for (let i = 0; i < T * semDim; i++) tanhS[i] = Math.tanh(S[i]);
      const proj = linear(tanhS, T, semDim, tr.ap_w, semDim, tr.ap_b);
      const sc = new Float32Array(T);
      let mx = -Infinity;
      for (let t = 0; t < T; t++) {
        let s = 0; const pb = t * semDim;
        for (let c = 0; c < semDim; c++) s += proj[pb + c] * tr.ap_q.get(c);
        s *= scoreScale; sc[t] = s; if (s > mx) mx = s;
      }
      let sum = 0;
      for (let t = 0; t < T; t++) { const e = Math.exp(sc[t] - mx); sc[t] = e; sum += e; }
      for (let t = 0; t < T; t++) { const a = sc[t] / sum, sb = t * semDim; for (let c = 0; c < semDim; c++) sv[c] += a * S[sb + c]; }
      layerNorm(sv, 1, semDim, tr.pn_w, tr.pn_b);
    } else {
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
    }

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
    const skinTone = options.skinTone ?? "default";
    return order.slice(0, Math.max(0, limit)).map((i) => ({
      emoji: applyEmojiSkinTone(labels[i], skinTone),
      confidence: logits[i] / sum,
    }));
  }
}

/** Creates an {@link EmoModel} from raw buffers (lowest-level entry). */
export function createEmo(buffers: { weights: Uint8Array; tokenizer: Uint8Array; meta: EmoMeta }): EmoModel {
  return new EmoModel(buffers.weights, buffers.tokenizer, buffers.meta);
}
