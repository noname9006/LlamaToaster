import { describe, expect, it } from "vitest";
import {
  detectThermalThrottle,
  SensorSampleBuffer,
  THROTTLE_MIN_REPEATS,
  THROTTLE_MIN_SAMPLES,
} from "./sensors.js";

describe("M6 thermal detection rule", () => {
  it("flags a sagging second half", () => {
    // ~2480 MHz across the first half, 1971 in the second -- the plan's own
    // worked screen-3 row.
    const detection = detectThermalThrottle({
      samples: [2480, 2479, 2481, 2478, 1971, 1975, 1969, 1972],
      repeats: 5,
      gpuTempCMax: 89,
    });
    expect(detection.throttled).toBe(true);
    expect(detection.ineligibleReason).toBeNull();
  });

  it("does not flag a steady clock", () => {
    const detection = detectThermalThrottle({
      samples: [2480, 2479, 2481, 2478, 2477, 2480],
      repeats: 5,
      gpuTempCMax: 78,
    });
    expect(detection.throttled).toBe(false);
  });

  it("flags exactly at the 0.95 boundary and not a hair above it", () => {
    const at = detectThermalThrottle({ samples: [100, 100, 95, 95], repeats: 4, gpuTempCMax: 70 });
    expect(at.throttled).toBe(true);
    const above = detectThermalThrottle({ samples: [100, 100, 96, 96], repeats: 4, gpuTempCMax: 70 });
    expect(above.throttled).toBe(false);
  });

  it("puts the extra sample in the FIRST half on an odd count, so the rule is exactly reproducible", () => {
    const detection = detectThermalThrottle({
      samples: [100, 100, 100, 90, 90],
      repeats: 4,
      gpuTempCMax: 70,
    });
    // first half = [100,100,100] (mean 100), second = [90,90] (mean 90).
    expect(detection.firstHalfMean).toBe(100);
    expect(detection.secondHalfMean).toBe(90);
    expect(detection.throttled).toBe(true);
  });

  it("is ineligible below 4 repeats, and says so rather than silently not flagging", () => {
    const detection = detectThermalThrottle({
      samples: [2480, 2480, 1000, 1000],
      repeats: 3,
      gpuTempCMax: 89,
    });
    expect(detection.throttled).toBe(false);
    expect(detection.ineligibleReason).toContain(String(THROTTLE_MIN_REPEATS));
  });

  it("is ineligible below 4 clock samples", () => {
    const detection = detectThermalThrottle({ samples: [2480, 1000], repeats: 8, gpuTempCMax: 89 });
    expect(detection.throttled).toBe(false);
    expect(detection.ineligibleReason).toContain(String(THROTTLE_MIN_SAMPLES));
  });

  it("never flags without a temperature reading -- a sensorless backend writes NULLs", () => {
    const detection = detectThermalThrottle({
      samples: [2480, 2480, 1000, 1000],
      repeats: 8,
      gpuTempCMax: null,
    });
    expect(detection.throttled).toBe(false);
    expect(detection.ineligibleReason).toContain("no temperature sensor");
  });
});

describe("M6 sample buffer (the worker owns every M6 column)", () => {
  it("derives clock min, temp max and the flag from its own samples", () => {
    const buffer = new SensorSampleBuffer();
    for (const [clockMhz, tempC] of [
      [2480, 70],
      [2479, 74],
      [2481, 80],
      [2478, 84],
      [1971, 88],
      [1975, 89],
      [1969, 89],
      [1972, 87],
    ] as const) {
      buffer.add({ clockMhz, tempC, source: "sensor_amdgpu_hwmon" });
    }
    const report = buffer.report(5);
    expect(report.gpu_temp_c_max).toBe(89);
    expect(report.gpu_clock_mhz_min).toBe(1969);
    expect(report.gpu_clock_samples).toHaveLength(8);
    expect(report.throttled).toBe(true);
    expect(buffer.source).toBe("sensor_amdgpu_hwmon");
  });

  it("a sensorless backend produces NULL columns and no flag", () => {
    const buffer = new SensorSampleBuffer();
    buffer.add({ clockMhz: null, tempC: null, source: null });
    const report = buffer.report(8);
    expect(report).toMatchObject({
      gpu_temp_c_max: null,
      gpu_clock_mhz_min: null,
      gpu_clock_samples: null,
      throttled: false,
    });
  });

  it("opening the timed window discards model-load idle clocks", () => {
    const buffer = new SensorSampleBuffer();
    // Idle clocks during model load would otherwise fabricate a "sag".
    buffer.add({ clockMhz: 300, tempC: 45, source: "sensor_nvidia_smi" });
    buffer.add({ clockMhz: 300, tempC: 45, source: "sensor_nvidia_smi" });
    buffer.openTimedWindow();
    for (const clockMhz of [2000, 2000, 1990, 1995]) {
      buffer.add({ clockMhz, tempC: 70, source: "sensor_nvidia_smi" });
    }
    const report = buffer.report(4);
    expect(report.gpu_clock_samples).toEqual([2000, 2000, 1990, 1995]);
    expect(report.gpu_clock_mhz_min).toBe(1990);
    expect(report.throttled).toBe(false);
  });
});
