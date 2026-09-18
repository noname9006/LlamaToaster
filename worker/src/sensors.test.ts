import { describe, expect, it } from "vitest";
import { SensorSampleBuffer, sampleFromLhmReadings } from "./sensors.js";

describe("M6 sample buffer (the worker owns every M6 column)", () => {
  it("derives clock min and temp max from its own samples", () => {
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
    const report = buffer.report();
    expect(report.gpu_temp_c_max).toBe(89);
    expect(report.gpu_clock_mhz_min).toBe(1969);
    expect(report.gpu_clock_samples).toHaveLength(8);
    expect(buffer.source).toBe("sensor_amdgpu_hwmon");
  });

  it("a sensorless backend produces NULL columns", () => {
    const buffer = new SensorSampleBuffer();
    buffer.add({ clockMhz: null, tempC: null, source: null });
    const report = buffer.report();
    expect(report).toMatchObject({
      gpu_temp_c_max: null,
      gpu_clock_mhz_min: null,
      gpu_clock_samples: null,
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
    const report = buffer.report();
    expect(report.gpu_clock_samples).toEqual([2000, 2000, 1990, 1995]);
    expect(report.gpu_clock_mhz_min).toBe(1990);
  });
});

// Windows + AMD/Intel fall back to LibreHardwareMonitor (readLhmSensors);
// sampleFromLhmReadings is its selection rule, so that's what's pinned here.
describe("M6 sensor_lhm selection rule (Windows LHM fallback)", () => {
  it("picks GPU-labelled readings and never promotes motherboard/CPU entries", () => {
    const sample = sampleFromLhmReadings([
      { label: "Temperature #1", kind: "temp", value: 42 }, // mainboard
      { label: "CPU Package", kind: "temp", value: 65 },
      { label: "Core #1", kind: "temp", value: 70 }, // CPU core
      { label: "Bus Speed", kind: "clock", value: 100 },
      { label: "GPU Hot Spot Temperature", kind: "temp", value: 84 },
      { label: "GPU Core", kind: "clock", value: 2480 },
    ]);
    // Sample came back (a usable GPU pair exists); these assert exactly what
    // got picked from the tree above.
    expect(sample).toMatchObject({ tempC: 84, clockMhz: 2480, source: "sensor_lhm" });
  });

  it("prefers the shader/core clock over other GPU-labelled clocks", () => {
    const sample = sampleFromLhmReadings([
      { label: "GPU Memory Clock", kind: "clock", value: 2000 },
      { label: "GPU Core", kind: "clock", value: 2480 },
      { label: "GPU SOC Clock", kind: "clock", value: 1000 },
    ]);
    expect(sample?.clockMhz).toBe(2480);
    expect(sample?.tempC).toBeNull(); // partial: clocks without temps still count
  });

  it("falls back to any GPU-labelled clock when no core-labelled one exists", () => {
    const sample = sampleFromLhmReadings([{ label: "GPU Memory Clock", kind: "clock", value: 2000 }]);
    expect(sample).toMatchObject({ clockMhz: 2000, tempC: null, source: "sensor_lhm" });
  });

  it("returns null when every reading is non-GPU -- declared unavailable, not guessed", () => {
    const sample = sampleFromLhmReadings([
      { label: "CPU Package", kind: "temp", value: 65 },
      { label: "Bus Speed", kind: "clock", value: 100 },
    ]);
    expect(sample).toBeNull();
  });
});
