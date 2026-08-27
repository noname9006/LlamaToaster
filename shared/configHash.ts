import type { EngineKind } from "./engineSpec.js";

export interface ConfigHashInput {
  n_prompt?: number;
  n_gen?: number;
  n_depth?: number;
  threads?: number;
  n_gpu_layers?: number;
  batch_size?: number;
  ubatch_size?: number;
  cache_type_k?: string;
  cache_type_v?: string;
  flash_attn?: string;
  n_gpu_layers_draft?: number;
  n_cpu_moe?: number;
  engine?: EngineKind;
  concurrency?: number;
}

const DEFAULTS: Record<string, number | string> = {
  n_prompt: 0,
  n_gen: 0,
  n_depth: 0,
  threads: 0,
  n_gpu_layers: 0,
  batch_size: 0,
  ubatch_size: 0,
  cache_type_k: "",
  cache_type_v: "",
  flash_attn: "",
  n_gpu_layers_draft: 0,
  n_cpu_moe: 0,
  engine: "bench",
  concurrency: 1,
};

export const CONFIG_HASH_INCLUDED_AXES = Object.keys(DEFAULTS);
export const CONFIG_HASH_EXCLUDED_AXES = ["spec", "spec_type", "spec_n_max", "spec_n_min"] as const;

function canonicalize(item: ConfigHashInput): Record<string, number | string> {
  const out: Record<string, number | string> = {};
  for (const axis of CONFIG_HASH_INCLUDED_AXES) {
    const raw = (item as Record<string, unknown>)[axis];
    if (typeof raw === "number" && Number.isFinite(raw)) {
      out[axis] = raw;
    } else if (typeof raw === "string") {
      out[axis] = raw;
    } else {
      out[axis] = DEFAULTS[axis];
    }
  }
  return out;
}

const HEX = "0123456789abcdef";

function sha256Hex(input: string): string {
  const msg = new TextEncoder().encode(input);
  const bitLen = msg.length * 8;
  const paddedLen = (((msg.length + 8) >> 6) + 1) << 6;
  const words = new Uint32Array(paddedLen >> 2);
  for (let i = 0; i < msg.length; i++) {
    words[i >> 2] |= msg[i] << ((3 - (i & 3)) * 8);
  }
  words[msg.length >> 2] |= 0x80 << ((3 - (msg.length & 3)) * 8);
  words[words.length - 1] = bitLen >>> 0;
  words[words.length - 2] = Math.floor(bitLen / 0x100000000);

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const k = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];

  const w = new Uint32Array(64);
  const rotr = (x: number, n: number) => (x >>> n) | (x << (32 - n));

  for (let block = 0; block < words.length; block += 16) {
    for (let t = 0; t < 16; t++) w[t] = words[block + t];
    for (let t = 16; t < 64; t++) {
      const s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >>> 3);
      const s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >>> 10);
      w[t] = (w[t - 16] + s0 + w[t - 7] + s1) >>> 0;
    }
    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;
    for (let t = 0; t < 64; t++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + k[t] + w[t]) >>> 0;
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }
    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
    h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0;
    h7 = (h7 + h) >>> 0;
  }

  let hex = "";
  const outs = [h0, h1, h2, h3, h4, h5, h6, h7];
  for (const word of outs) {
    for (let shift = 28; shift >= 0; shift -= 4) {
      hex += HEX[(word >>> shift) & 0xf];
    }
  }
  return hex;
}

export function configHash(item: ConfigHashInput): string {
  const canonical = canonicalize(item);
  const keys = Object.keys(canonical).sort();
  const stable = keys.map((k) => `${JSON.stringify(k)}:${JSON.stringify(canonical[k])}`).join(",");
  return sha256Hex(stable);
}

// N2 -- the placement an affordability estimate assumed, hashed so a
// verification row is bound to it: verification is per (machine, build, KV
// pair, placement) and changing any of them invalidates the verdict.
export interface PlacementInput {
  ngl: number;
  n_cpu_moe?: number;
  slots?: number;
}

export function placementHash(p: PlacementInput): string {
  const canonical = {
    ngl: Math.trunc(p.ngl),
    n_cpu_moe: Math.trunc(p.n_cpu_moe ?? 0),
    slots: Math.trunc(p.slots ?? 1),
  };
  const keys = Object.keys(canonical).sort() as (keyof typeof canonical)[];
  return sha256Hex(keys.map((k) => `${JSON.stringify(k)}:${JSON.stringify(canonical[k])}`).join(","));
}
