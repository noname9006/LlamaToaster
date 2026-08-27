// BENCHMARKING_PLAN_V8.md M6 -- adapter clock/temperature telemetry, read on
// the same cadence and with the same vendor-split dispatch as the VRAM
// reader (worker/src/vram.ts). There is no cross-platform GPU sensor source,
// and systeminformation's own fields were already proven unusable on the
// reference box for memory, so each platform/vendor pair gets the source the
// plan names for it -- and every pair the plan leaves out reports NULL with
// its availability declared up front, never a silently-missing flag later.

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readdir, readFile } from "node:fs/promises";
import { platform as osPlatform } from "node:os";
import type { Backend, SensorMeasurementSource } from "../../shared/types.js";

const execFileAsync = promisify(execFile);

// Same budget the VRAM readers use -- nvidia-smi/rocm-smi execs cost
// ~200-500 ms, which fits comfortably inside the ~6 s VRAM cadence.
const EXEC_TIMEOUT_MS = 10_000;

const DRM_CARD_DIR = "/sys/class/drm";
const CARD_DIR_RE = /^card\d+$/;

export interface SensorSample {
  /** Lowest-level adapter shader clock, MHz. Null where no source exposes it. */
  clockMhz: number | null;
  /** Adapter peak temperature, °C. Null where no source exposes it. */
  tempC: number | null;
  /** Which source produced this sample -- worst-source tracked across an item. */
  source: SensorMeasurementSource | null;
}

export interface SensorAvailability {
  clock: boolean;
  temp: boolean;
  source: SensorMeasurementSource | null;
}

const NO_SAMPLE: SensorSample = { clockMhz: null, tempC: null, source: null };

export async function readGpuSensors(backend: Backend): Promise<SensorSample> {
  try {
    if (backend === "cpu") return NO_SAMPLE;
    if (backend === "cuda") return (await readNvidiaSmiSensors()) ?? NO_SAMPLE;
    const plat = osPlatform();
    if (plat === "linux") {
      // ROCm's canonical CLI first, then the amdgpu hwmon attributes, which
      // remain valid under ROCm kernels too -- and are the only free source
      // on a stock-driver vulkan box.
      if (backend === "rocm") {
        const rocm = await readRocmSmiSensors();
        if (rocm) return rocm;
      }
      const hwmon = await readAmdgpuHwmonSensors();
      if (hwmon) return hwmon;
      // An NVIDIA card running the vulkan backend still has nvidia-smi.
      return (await readNvidiaSmiSensors()) ?? NO_SAMPLE;
    }
    if (plat === "win32") {
      // NVIDIA ships nvidia-smi on Windows too -- exact vendor CLI first, the
      // same priority order every other branch uses. For AMD/Intel on Windows
      // there is no vendor-agnostic CLI (WMI thermal zones are
      // motherboard-level, PDH has no sensor counters), so the fallback is
      // LibreHardwareMonitor (readLhmSensors below) and NULL with declared
      // availability beyond that -- guessed motherboards temps were never
      // acceptable, but a real adapter reading via LHM is.
      return (await readNvidiaSmiSensors()) ?? (await readLhmSensors()) ?? NO_SAMPLE;
    }
    // Apple metal: powermetrics needs sudo. Declared unavailable in v1.
    return NO_SAMPLE;
  } catch {
    return NO_SAMPLE;
  }
}

// Declared on the heartbeat so a machine card can say "clock · temp
// available" up front -- a later thermally_throttled flag must never
// surprise a machine that could not have produced one.
export async function detectSensorAvailability(backend: Backend): Promise<SensorAvailability> {
  const sample = await readGpuSensors(backend);
  return {
    clock: sample.clockMhz != null,
    temp: sample.tempC != null,
    source: sample.source,
  };
}

async function readNvidiaSmiSensors(): Promise<SensorSample | null> {
  try {
    const { stdout } = await execFileAsync(
      "nvidia-smi",
      ["--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader,nounits"],
      { timeout: EXEC_TIMEOUT_MS }
    );
    const line = stdout.split("\n").find((l) => l.trim().length > 0);
    if (!line) return null;
    const [clockRaw, tempRaw] = line.split(",").map((p) => p.trim());
    const clockMhz = toFiniteNumber(clockRaw);
    const tempC = toFiniteNumber(tempRaw);
    if (clockMhz == null && tempC == null) return null;
    return { clockMhz, tempC, source: "sensor_nvidia_smi" };
  } catch {
    return null;
  }
}

async function readRocmSmiSensors(): Promise<SensorSample | null> {
  try {
    const { stdout } = await execFileAsync("rocm-smi", ["--showtemp", "--showclocks", "--json"], {
      timeout: EXEC_TIMEOUT_MS,
    });
    const parsed = JSON.parse(stdout) as Record<string, Record<string, string>>;
    for (const card of Object.values(parsed)) {
      const sample = sensorsFromRocmCard(card);
      if (sample) return sample;
    }
    return null;
  } catch {
    // Fall back to text parsing -- older rocm-smi builds have no --json.
    return readRocmSmiSensorsText();
  }
}

// rocm-smi's JSON keys carry their units in the key name and vary by
// version ("Temperature (Sensor edge) (C)", "sclk clock speed:"), so match
// on shape rather than an exact key.
function sensorsFromRocmCard(card: Record<string, string>): SensorSample | null {
  let tempC: number | null = null;
  let clockMhz: number | null = null;
  for (const [key, rawValue] of Object.entries(card)) {
    const k = key.toLowerCase();
    const value = String(rawValue);
    if (tempC == null && k.includes("temperature")) {
      tempC = toFiniteNumber(value.replace(/[^0-9.\-]/g, ""));
    }
    if (clockMhz == null && k.includes("sclk")) {
      const mhz = /(\d+)\s*mhz/i.exec(value);
      clockMhz = mhz ? toFiniteNumber(mhz[1]) : toFiniteNumber(value.replace(/[^0-9.]/g, ""));
    }
  }
  if (tempC == null && clockMhz == null) return null;
  return { tempC, clockMhz, source: "sensor_rocm_smi" };
}

async function readRocmSmiSensorsText(): Promise<SensorSample | null> {
  try {
    const { stdout } = await execFileAsync("rocm-smi", ["--showtemp", "--showclocks"], {
      timeout: EXEC_TIMEOUT_MS,
    });
    const temp = /Temperature[^\n]*?:\s*([0-9.]+)/i.exec(stdout);
    const sclk = /sclk[^\n]*?\(?\s*(\d+)\s*Mhz/i.exec(stdout);
    const tempC = temp ? toFiniteNumber(temp[1]) : null;
    const clockMhz = sclk ? toFiniteNumber(sclk[1]) : null;
    if (tempC == null && clockMhz == null) return null;
    return { tempC, clockMhz, source: "sensor_rocm_smi" };
  } catch {
    return null;
  }
}

// amdgpu hwmon: temp1_input is millidegrees Celsius; freq1_input is Hz (not
// mHz -- the kernel's own amdgpu thermal documentation). Free, no install,
// and the same interface class the VRAM reader already cites.
async function readAmdgpuHwmonSensors(): Promise<SensorSample | null> {
  try {
    const cards = await readdir(DRM_CARD_DIR);
    for (const card of cards) {
      if (!CARD_DIR_RE.test(card)) continue;
      const hwmonBase = `${DRM_CARD_DIR}/${card}/device/hwmon`;
      let hwmons: string[];
      try {
        hwmons = await readdir(hwmonBase);
      } catch {
        continue;
      }
      for (const hwmon of hwmons) {
        const dir = `${hwmonBase}/${hwmon}`;
        const tempC = await readNumberFile(`${dir}/temp1_input`, 1000);
        const clockMhz = await readNumberFile(`${dir}/freq1_input`, 1_000_000);
        if (tempC != null || clockMhz != null) {
          return { tempC, clockMhz, source: "sensor_amdgpu_hwmon" };
        }
      }
    }
    return null;
  } catch {
    return null;
  }
}

async function readNumberFile(path: string, divisor: number): Promise<number | null> {
  try {
    const raw = await readFile(path, "utf8");
    const value = Number(raw.trim());
    if (!Number.isFinite(value)) return null;
    return Math.round(value / divisor);
  } catch {
    return null;
  }
}

function toFiniteNumber(raw: string | undefined): number | null {
  if (raw == null) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

// --- Windows-only LibreHardwareMonitor (sensor_lhm) --------------------------
//
// sensor_lhm has been reserved in SENSOR_MEASUREMENT_SOURCES for exactly the
// pair this section serves: Windows + AMD/Intel graphics, where there is no
// vendor CLI the way nvidia-smi/rocm-smi exist elsewhere (WMI thermal zones
// are motherboard-level, PDH has no GPU sensor counters). LibreHardwareMonitor
// reaches those adapters through their driver interfaces directly and exposes
// two machine-readable views, tried here in cost order:
//
//   1. Its web server's /data.json tree (the feed its own UI renders from):
//      one HTTP GET, far cheaper than anything shelled at the sampler's 2 s
//      cadence. Requires "Remote Web Server" ticked in the app's options.
//   2. Its WMI namespace root\LibreHardwareMonitor: works whenever the app is
//      merely RUNNING, with no setting toggled, at the price of a ~1-3 s
//      PowerShell spawn per read -- well inside EXEC_TIMEOUT_MS, just laggy.
//
// Either way the app itself must be running AND elevated: without admin its
// GPU drivers are no more reachable from us than they are from anyone else.
// The HTTP port is overridable via LHM_HTTP_PORT (default 8085).

// Exported so the worker's startup log can cite the real port (LHM_HTTP_PORT
// override) in its Windows AMD/Intel LibreHardwareMonitor guidance.
export const LHM_HTTP_PORT = Number(process.env.LHM_HTTP_PORT ?? 8085);
const LHM_HTTP_URL = `http://127.0.0.1:${LHM_HTTP_PORT}/data.json`;

/** One candidate reading harvested from an LHM surface, pre-selection. */
export interface LhmReading {
  label: string;
  kind: "temp" | "clock";
  value: number;
}

export async function readLhmSensors(): Promise<SensorSample | null> {
  return (await readLhmHttpSensors()) ?? (await readLhmWmiSensors());
}

// /data.json is the app's whole node tree: hardware nodes carry children, and
// leaf values come with their UI icon (ImageURL) naming the sensor class --
// images/temperature.png, images/clock.png and friends. Match the icon rather
// than the display text, which varies by vendor/version exactly the way
// rocm-smi's JSON keys did.
interface LhmNode {
  Text?: string;
  Value?: string;
  ImageURL?: string;
  Children?: LhmNode[];
}

async function readLhmHttpSensors(): Promise<SensorSample | null> {
  try {
    const res = await fetch(LHM_HTTP_URL, { signal: AbortSignal.timeout(EXEC_TIMEOUT_MS) });
    if (!res.ok) return null;
    const root = (await res.json()) as LhmNode;
    const readings: LhmReading[] = [];
    collectLhmReadings(root, readings);
    return sampleFromLhmReadings(readings);
  } catch {
    return null;
  }
}

function collectLhmReadings(node: LhmNode, out: LhmReading[]): void {
  const url = node.ImageURL ?? "";
  const value = parseUnitValue(node.Value);
  if (value != null && /temperature/i.test(url)) {
    out.push({ label: node.Text ?? "", kind: "temp", value });
  } else if (value != null && /clock/i.test(url)) {
    out.push({ label: node.Text ?? "", kind: "clock", value });
  }
  for (const child of node.Children ?? []) collectLhmReadings(child, out);
}

// Selection rule, stated so it stays reproducible across app versions whose
// sensor labels drift: a reading only counts when its OWN label says GPU
// ("GPU Core", "GPU Hot Spot Temperature", ...) -- CPU/motherboard/mainboard
// entries in the same tree are never promoted into adapter numbers, matching
// this file's standing "declared rather than guessed" posture. Among
// GPU-labelled candidates the first temperature wins; clocks prefer a core /
// shader-clock label (what the M6 sag rule actually tracks) before falling
// back to any other GPU-labelled clock (e.g. memory).
export function sampleFromLhmReadings(readings: LhmReading[]): SensorSample | null {
  const gpuTemps = readings.filter((r) => r.kind === "temp" && /gpu/i.test(r.label));
  const gpuClocks = readings.filter((r) => r.kind === "clock" && /gpu/i.test(r.label));
  const clock = gpuClocks.find((r) => /gpu\s*core/i.test(r.label)) ?? gpuClocks[0];
  const tempC = gpuTemps[0]?.value ?? null;
  const clockMhz = clock?.value ?? null;
  if (tempC == null && clockMhz == null) return null;
  return { tempC, clockMhz, source: "sensor_lhm" };
}

// Values arrive rendered with units ("45 °C", "1600 MHz"); same strip-and-parse
// rocm text parsing does.
function parseUnitValue(raw: string | undefined): number | null {
  if (!raw) return null;
  return toFiniteNumber(raw.replace(/[^0-9.\-]/g, ""));
}

// WMI fallback: typed columns instead of icon strings. Get-CimInstance filters
// to GPU-classed hardware, joins Sensors onto it by Identifier/Parent, and
// keeps only the two classes this file maps. ConvertTo-Json emits one object
// for a single row and an array otherwise, hence the both-shapes parse.
interface LhmWmiSensorRow {
  Name?: unknown;
  SensorType?: unknown;
  Value?: unknown;
}

const LHM_WMI_SCRIPT = [
  "$ErrorActionPreference='Stop'",
  "$hw=@(Get-CimInstance -Namespace root/LibreHardwareMonitor -ClassName Hardware | Where-Object { $_.HardwareType -eq 'GPU' })",
  "$s=Get-CimInstance -Namespace root/LibreHardwareMonitor -ClassName Sensor | Where-Object { $hw.Identifier -contains $_.Parent -and ($_.SensorType -eq 'Temperature' -or $_.SensorType -eq 'Clock') }",
  "$s | Select-Object Name,SensorType,Value | ConvertTo-Json -Compress",
].join("; ");

async function readLhmWmiSensors(): Promise<SensorSample | null> {
  try {
    const { stdout } = await execFileAsync(
      "powershell.exe",
      ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", LHM_WMI_SCRIPT],
      { timeout: EXEC_TIMEOUT_MS }
    );
    const trimmed = stdout.trim();
    if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return null;
    const parsed = JSON.parse(trimmed) as LhmWmiSensorRow | LhmWmiSensorRow[];
    const rows = Array.isArray(parsed) ? parsed : [parsed];
    const readings: LhmReading[] = [];
    for (const row of rows) {
      const kindRaw = String(row.SensorType ?? "");
      const kind = kindRaw === "Temperature" ? "temp" : kindRaw === "Clock" ? "clock" : null;
      if (!kind || typeof row.Name !== "string") continue;
      const value = wmiNumeric(row.Value);
      if (value != null) readings.push({ label: row.Name, kind, value });
    }
    return sampleFromLhmReadings(readings);
  } catch {
    return null;
  }
}

// Older OpenHardwareMonitor-lineage builds publish Value as a float ARRAY of
// per-tick readings instead of a scalar; the last element is the current one.
function wmiNumeric(raw: unknown): number | null {
  if (Array.isArray(raw)) {
    raw = raw[raw.length - 1];
  }
  return typeof raw === "number" && Number.isFinite(raw) ? raw : null;
}

// --- M6 detection rule (computable, deterministic) --------------------------

export interface ThrottleDetectionInput {
  /** Clock samples in chronological order, sampled on the VRAM cadence. */
  samples: number[];
  /** Repeats the item ran -- the flag needs >= 4 so each half holds >= 2. */
  repeats: number;
  /** Adapter peak temperature; the flag requires a real temperature reading. */
  gpuTempCMax: number | null;
}

export interface ThrottleDetection {
  throttled: boolean;
  /** Why the item was ineligible, when it was -- stated, never silent. */
  ineligibleReason: string | null;
  firstHalfMean: number | null;
  secondHalfMean: number | null;
}

export const THROTTLE_MIN_REPEATS = 4;
export const THROTTLE_MIN_SAMPLES = 4;
export const THROTTLE_SAG_RATIO = 0.95;

// Split the sample array chronologically in half -- with uniform cadence
// this approximates repeat halves. An odd sample count puts the extras in
// the FIRST half (floor split), stated so the rule stays exactly
// reproducible. Flag when mean(secondHalf) <= 0.95 * mean(firstHalf) AND a
// temperature reading exists. The flagged row is kept, not failed:
// burst-vs-sustained is visible data.
export function detectThermalThrottle(input: ThrottleDetectionInput): ThrottleDetection {
  const { samples, repeats, gpuTempCMax } = input;
  if (repeats < THROTTLE_MIN_REPEATS) {
    return {
      throttled: false,
      ineligibleReason: `needs at least ${THROTTLE_MIN_REPEATS} repeats (this item ran ${repeats})`,
      firstHalfMean: null,
      secondHalfMean: null,
    };
  }
  if (samples.length < THROTTLE_MIN_SAMPLES) {
    return {
      throttled: false,
      ineligibleReason: `needs at least ${THROTTLE_MIN_SAMPLES} clock samples (this item produced ${samples.length})`,
      firstHalfMean: null,
      secondHalfMean: null,
    };
  }
  const secondHalfLength = Math.floor(samples.length / 2);
  const firstHalf = samples.slice(0, samples.length - secondHalfLength);
  const secondHalf = samples.slice(samples.length - secondHalfLength);
  const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
  const firstHalfMean = mean(firstHalf);
  const secondHalfMean = mean(secondHalf);
  if (gpuTempCMax == null) {
    return {
      throttled: false,
      ineligibleReason: "no temperature sensor on this backend, so a clock sag cannot be attributed to heat",
      firstHalfMean,
      secondHalfMean,
    };
  }
  return {
    throttled: secondHalfMean <= THROTTLE_SAG_RATIO * firstHalfMean,
    ineligibleReason: null,
    firstHalfMean,
    secondHalfMean,
  };
}

// --- M6 sample buffer -------------------------------------------------------

// The worker owns every M6 column: it derives the flag, gpu_clock_mhz_min and
// gpu_temp_c_max at item end from its own sample buffer, and the server
// stores them verbatim without ever re-deriving. One writer per column.
export class SensorSampleBuffer {
  private clockSamples: number[] = [];
  private tempMax: number | null = null;
  private worstSource: SensorMeasurementSource | null = null;
  /** Samples taken before the timed window opened, discarded on open(). */
  private windowOpen = true;

  reset(): void {
    this.clockSamples = [];
    this.tempMax = null;
    this.worstSource = null;
    this.windowOpen = true;
  }

  // Server items bracket sampling from the first timed request through
  // process exit: model-load idle clocks inside the sampled span would
  // otherwise mask or fabricate sag. llama-bench items approximate with the
  // whole item, since repeats are binary-internal -- they simply never call
  // this.
  openTimedWindow(): void {
    this.clockSamples = [];
    this.tempMax = null;
    this.windowOpen = true;
  }

  add(sample: SensorSample): void {
    if (!this.windowOpen) return;
    if (sample.clockMhz != null) this.clockSamples.push(sample.clockMhz);
    if (sample.tempC != null) this.tempMax = Math.max(this.tempMax ?? Number.NEGATIVE_INFINITY, sample.tempC);
    if (sample.source != null && this.worstSource == null) this.worstSource = sample.source;
  }

  get samples(): number[] {
    return [...this.clockSamples];
  }

  get source(): SensorMeasurementSource | null {
    return this.worstSource;
  }

  report(repeats: number): {
    gpu_temp_c_max: number | null;
    gpu_clock_mhz_min: number | null;
    gpu_clock_samples: number[] | null;
    throttled: boolean;
    detection: ThrottleDetection;
  } {
    const detection = detectThermalThrottle({
      samples: this.clockSamples,
      repeats,
      gpuTempCMax: this.tempMax,
    });
    return {
      gpu_temp_c_max: this.tempMax,
      gpu_clock_mhz_min: this.clockSamples.length > 0 ? Math.min(...this.clockSamples) : null,
      gpu_clock_samples: this.clockSamples.length > 0 ? [...this.clockSamples] : null,
      throttled: detection.throttled,
      detection,
    };
  }
}
