// What the served llama-server says about itself on stderr. It runs at
// --verbosity 4: buffer sizes, offload summary and prompt-cache state appear there,
// and in a spike a marker in the prompt never did. Level 5 echoes request bodies,
// so nothing here may ever raise the level.
//
// Level 4 is not silent about requests, though: grammar and JSON-schema errors repeat
// what the client sent. Only the fields below are kept, plus a small ring of lines
// that passes through safeForRing, which goes to this machine's own log after a crash
// and nowhere else.

const BUFFER_LINE = /:\s*(\S+)\s+(model|KV|RS|compute|output)\s+buffer size\s*=\s*([\d.]+)\s*MiB/;
const OFFLOAD_LINE = /load_tensors:\s*offloaded\s+(\d+)\/(\d+)\s+layers/;
const SLOTS_LINE = /n_slots\s*=\s*(\d+),\s*n_ctx_slot\s*=\s*(\d+)/;
const CACHE_STATE_LINE = /cache state:\s*\d+\s+prompts?,\s*([\d.]+)\s*MiB/;
// The buffers on record stop being true: a new load in this process, or going to
// sleep, which frees them (measured: the process drops to 3 MiB of VRAM).
const LOAD_START_LINE = /load_model:\s*loading model|(entering|exiting) sleeping state/;
// Local buffers are host memory whatever backend made them.
const HOST_BUFFER_NAME = /^(CPU|CPU_\w+|\w+_Host)$/;
// A line worth showing when a load fails. Load-time only: no request has reached
// the process yet, so no prompt can be in it.
const LOAD_ERROR_LINE = /\sE\s|error|failed|out of memory|GGML_ASSERT|unable to|not enough|invalid/i;
const RING_LINES = 200;
const MAX_LOAD_ERRORS = 8;
// Timestamped llama.cpp log lines. Anything else is free text: grammar dumps, sampler
// parameter blocks, or a crash message.
const LOG_LINE = /^\d+\.\d+\.\d+\.\d+ [A-Z] /;
// Measured to repeat request content at level 4.
const ECHOES_REQUEST = /got exception|send_error|grammar|schema|parse:/i;
// What a crash prints with no timestamp, and the reason to keep a tail at all.
const CRASH_TEXT = /GGML_ASSERT|out of memory|bad_alloc|DeviceLost|device lost|vk::|segmentation|terminate called/i;
const BIND_FAILURE = /couldn't bind|address already in use/i;

export function safeForRing(line: string): boolean {
  if (CRASH_TEXT.test(line)) return true;
  return LOG_LINE.test(line) && !ECHOES_REQUEST.test(line);
}

export interface LoadReport {
  weightsGpuMib: number | null;
  kvGpuMib: number | null;
  computeGpuMib: number | null;
  weightsHostMib: number | null;
  computeHostMib: number | null;
  promptCacheMib: number | null;
  gpuLayers: number | null;
  totalLayers: number | null;
  nSlots: number | null;
  nCtxSlot: number | null;
  /** Buffer lines of the current load, in the form bench.ts's parseGpuBufferReport reads. */
  bufferLines: string[];
}

export class DaemonLog {
  private partial = "";
  private ring: string[] = [];
  private modelMib = new Map<string, number>();
  private kvRun = new Map<string, number>();
  private inKvRun = false;
  private latestMib = new Map<string, number>();
  private promptCacheMib: number | null = null;
  private gpuLayers: number | null = null;
  private totalLayers: number | null = null;
  private nSlots: number | null = null;
  private nCtxSlot: number | null = null;
  private bufferLines: string[] = [];
  private loadErrors: string[] = [];
  private loading = true;
  private bindFailure = false;

  /** The private port was taken between choosing it and llama-server binding it. */
  get bindFailed(): boolean {
    return this.bindFailure;
  }

  /** Requests can arrive from here on, so stop collecting lines for a load error. */
  markReady(): void {
    this.loading = false;
    this.loadErrors = [];
  }

  feed(chunk: Buffer | string): void {
    this.partial += chunk.toString();
    const lines = this.partial.split(/\r?\n/);
    this.partial = lines.pop() ?? "";
    // A pathological line with no newline must not grow without bound.
    if (this.partial.length > 64_000) this.partial = "";
    for (const line of lines) this.line(line);
  }

  private line(line: string): void {
    if (safeForRing(line)) {
      this.ring.push(line.length > 400 ? `${line.slice(0, 400)}...` : line);
      if (this.ring.length > RING_LINES) this.ring.shift();
    }

    if (BIND_FAILURE.test(line)) this.bindFailure = true;
    if (LOAD_START_LINE.test(line)) this.resetLoad();
    const buffer = BUFFER_LINE.exec(line);
    if (buffer) {
      this.buffer(buffer[1], buffer[2], Number(buffer[3]));
      if (this.bufferLines.length < 64) this.bufferLines.push(line);
      return;
    }
    const offload = OFFLOAD_LINE.exec(line);
    if (offload) {
      this.gpuLayers = Number(offload[1]);
      this.totalLayers = Number(offload[2]);
      return;
    }
    const slots = SLOTS_LINE.exec(line);
    if (slots) {
      this.nSlots = Number(slots[1]);
      this.nCtxSlot = Number(slots[2]);
      return;
    }
    const cache = CACHE_STATE_LINE.exec(line);
    if (cache) {
      this.promptCacheMib = Number(cache[1]);
      return;
    }
    if (this.loading && this.loadErrors.length < MAX_LOAD_ERRORS && LOAD_ERROR_LINE.test(line)) {
      this.loadErrors.push(line.replace(/^[\d.:]+\s+\w\s+/, "").slice(0, 240));
    }
  }

  private resetLoad(): void {
    this.modelMib.clear();
    this.kvRun.clear();
    this.latestMib.clear();
    this.inKvRun = false;
    this.promptCacheMib = null;
    this.bufferLines = [];
  }

  private buffer(device: string, kind: string, mib: number): void {
    if (!Number.isFinite(mib)) return;
    const isCache = kind === "KV" || kind === "RS";
    // One context allocates several KV buffers in a row (full attention, then the
    // window); a later reservation replaces the whole run rather than adding to it.
    if (isCache && !this.inKvRun) this.kvRun.clear();
    this.inKvRun = isCache;
    if (kind === "model") this.modelMib.set(device, (this.modelMib.get(device) ?? 0) + mib);
    else if (isCache) this.kvRun.set(device, (this.kvRun.get(device) ?? 0) + mib);
    else this.latestMib.set(`${device}:${kind}`, mib);
  }

  report(): LoadReport {
    let weightsGpu = 0;
    let weightsHost = 0;
    let kvGpu = 0;
    let computeGpu = 0;
    let computeHost = 0;
    let seen = false;
    for (const [device, mib] of this.modelMib) {
      seen = true;
      if (HOST_BUFFER_NAME.test(device)) weightsHost += mib;
      else weightsGpu += mib;
    }
    for (const [device, mib] of this.kvRun) {
      seen = true;
      // A KV cache on the CPU counts toward the host's compute-side memory.
      if (HOST_BUFFER_NAME.test(device)) computeHost += mib;
      else kvGpu += mib;
    }
    for (const [key, mib] of this.latestMib) {
      seen = true;
      if (HOST_BUFFER_NAME.test(key.split(":")[0])) computeHost += mib;
      else computeGpu += mib;
    }
    return {
      weightsGpuMib: seen ? weightsGpu : null,
      kvGpuMib: seen ? kvGpu : null,
      computeGpuMib: seen ? computeGpu : null,
      weightsHostMib: seen ? weightsHost : null,
      computeHostMib: seen ? computeHost : null,
      promptCacheMib: this.promptCacheMib,
      gpuLayers: this.gpuLayers,
      totalLayers: this.totalLayers,
      nSlots: this.nSlots,
      nCtxSlot: this.nCtxSlot,
      bufferLines: [...this.bufferLines],
    };
  }

  /** Load-time problems, for a failed load's message. */
  loadErrorSummary(): string | null {
    return this.loadErrors.length > 0 ? this.loadErrors.join("\n").slice(0, 1500) : null;
  }

  /** The last lines, for this machine's own log after a crash. Never sent to the server. */
  tail(lines = 30): string {
    return this.ring.slice(-lines).join("\n");
  }
}
