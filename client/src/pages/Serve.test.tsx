// Audit 2026-11-01 (docs/AUDIT_REPORT_SERVE_2026-11-01.md): real renders of the page
// and its panels. Each of these failed against the code before the audit's fixes.
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { SamplingPanel } from "../components/serve/SettingsPanels";
import { UsagePanel } from "../components/serve/UsagePanel";
import { MemoryPanel } from "../components/serve/MemoryPanel";
import { ConnectionPanel } from "../components/serve/ConnectionPanel";
import { summarizeUsage } from "../../../shared/daemonUsage";
import type { SamplingSettings } from "../../../shared/daemonSettings";
import type { DaemonMemorySnapshot, DaemonStatus, DaemonUsageView, DaemonView } from "../../../shared/daemonTypes";

const mocks = vi.hoisted(() => ({
  getDaemon: vi.fn(),
  listModels: vi.fn(),
  getDaemonModelSettings: vi.fn(),
  getDaemonUsage: vi.fn(),
  unloadDaemonModel: vi.fn(),
  loadDaemonModel: vi.fn(),
  createDaemonKey: vi.fn(),
  saveDaemonSampling: vi.fn(),
}));
vi.mock("../api/client", () => ({ api: mocks, ApiError: class extends Error {} }));
const workers = [
  { id: "w1", displayName: "One", status: "idle", capabilities: ["daemon-v1"], hardware: { cpu: { cores: 8 } }, modelFiles: [{ path: "alpha.gguf", sha256: "a" }, { path: "zeta.gguf", sha256: "z" }, { path: "mmproj-x.gguf" }] },
];
vi.mock("../api/useWorkerStatus", () => ({ useWorkerStatuses: () => ({ workers, loaded: true }) }));
vi.mock("../components/Chart", () => ({ Chart: () => null }));
import { Serve } from "./Serve";

const models = [
  { id: "m-alpha", source: "local", filename: "alpha.gguf" },
  { id: "m-zeta", source: "local", filename: "zeta.gguf" },
];

const status = (over: Partial<DaemonStatus> = {}): DaemonStatus => ({
  phase: "ready",
  message: null,
  model_id: "m-zeta",
  model_name: "zeta",
  alias: "zeta",
  build_tag: "b1",
  load_hash: "h1",
  settings: { ctx_size: 2222 },
  mmproj_filename: null,
  draft_model_id: null,
  actual: null,
  loaded_at: 1,
  in_flight: 0,
  requests_total: 0,
  generation_defaults: null,
  gateway: { listening: true, host: "127.0.0.1", port: 8642, error: null },
  interfaces: [],
  active_build_tag: "b1",
  supported_settings: [],
  restarts: 0,
  usage_dropped: 0,
  autostart_model_id: null,
  memory: null,
  ...over,
});
const gone = { model_id: null, alias: null, load_hash: null, settings: null } as const;
const view = (s: DaemonStatus | null): DaemonView => ({ status: s, reported_at: Date.now(), capable: true, gateway: { host: "127.0.0.1", port: 8642, cors_origins: [], autostart: false }, keys: [], pending_job: null, last_error: null });
const usage = (): DaemonUsageView => ({
  loads: [
    { load_hash: "h1", model_id: "m", build_tag: "b1", settings: {}, actual: null, first_seen: 1, requests: 0 },
    { load_hash: "old", model_id: "m", build_tag: "b1", settings: {}, actual: null, first_seen: 0, requests: 50 },
  ],
  profile: summarizeUsage([]),
  curve: [],
  points: [],
  concurrent_excluded: 0,
  window_days: 30,
});

const nextPoll = () => act(async () => void (await new Promise((r) => setTimeout(r, 3300))));
const modelSelect = () => screen.getByLabelText("Model") as HTMLSelectElement;
const contextField = () => screen.getByLabelText("Context length") as HTMLInputElement;

function renderPage() {
  return render(
    <MemoryRouter>
      <Serve />
    </MemoryRouter>
  );
}

beforeEach(() => {
  localStorage.clear();
  mocks.listModels.mockResolvedValue(models);
  mocks.getDaemonModelSettings.mockImplementation(async (_w: string, m: string) => ({
    settings: { settings: { ctx_size: m === "m-zeta" ? 2222 : 1111 }, sampling: { values: {} }, alias: null, mmproj_filename: null, draft_model_id: null },
  }));
  mocks.getDaemonUsage.mockResolvedValue(usage());
  mocks.getDaemon.mockResolvedValue(view(status()));
  mocks.unloadDaemonModel.mockResolvedValue({ ok: true, job_id: "j" });
  mocks.loadDaemonModel.mockResolvedValue({ ok: true, job_id: "j" });
  Object.defineProperty(document, "visibilityState", { value: "visible", configurable: true });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("the selected model", () => {
  it.each([
    ["an unload", { phase: "idle", ...gone }],
    ["a failed load", { phase: "failed", message: "out of memory", ...gone }],
  ] as const)("stays put after %s, with its settings", async (_name, after) => {
    renderPage();
    await waitFor(() => expect(modelSelect().value).toBe("m-zeta"));
    await waitFor(() => expect(contextField().value).toBe("2222"));
    mocks.getDaemon.mockResolvedValue(view(status(after)));
    await nextPoll();
    expect(modelSelect().value).toBe("m-zeta");
    expect(contextField().value).toBe("2222");
    expect(screen.getAllByText("Load model").length).toBeGreaterThan(0);
  }, 10_000);

  it("keeps a model picked by hand when another is running", async () => {
    renderPage();
    await waitFor(() => expect(modelSelect().value).toBe("m-zeta"));
    fireEvent.change(modelSelect(), { target: { value: "m-alpha" } });
    await nextPoll();
    expect(modelSelect().value).toBe("m-alpha");
  }, 10_000);
});

describe("calling off a model", () => {
  it.each(["suspended", "crashed"] as const)("offers Unload while %s", async (phase) => {
    mocks.getDaemon.mockResolvedValue(view(status({ phase, load_hash: null })));
    renderPage();
    const button = await screen.findByText("Unload");
    fireEvent.click(button);
    await waitFor(() => expect(mocks.unloadDaemonModel).toHaveBeenCalledWith("w1"));
  });
});

describe("offering a reload", () => {
  it("counts a change of projector", async () => {
    renderPage();
    await waitFor(() => expect(contextField().value).toBe("2222"));
    expect(screen.queryByText(/differ from what is running/)).toBeNull();
    fireEvent.change(screen.getByLabelText("Projector file"), { target: { value: "mmproj-x.gguf" } });
    await screen.findByText(/differ from what is running/);
  });
});

describe("a hidden tab", () => {
  it("stops telling the machine someone is watching, and catches up when shown", async () => {
    renderPage();
    await waitFor(() => expect(mocks.getDaemon).toHaveBeenCalled());
    Object.defineProperty(document, "visibilityState", { value: "hidden", configurable: true });
    await nextPoll();
    mocks.getDaemon.mockClear();
    await nextPoll();
    expect(mocks.getDaemon).not.toHaveBeenCalled();
    Object.defineProperty(document, "visibilityState", { value: "visible", configurable: true });
    fireEvent(document, new Event("visibilitychange"));
    await waitFor(() => expect(mocks.getDaemon).toHaveBeenCalled());
  }, 12_000);
});

describe("stop strings", () => {
  function Stateful() {
    const [v, setV] = useState<SamplingSettings>({ values: {} });
    return (
      <>
        <SamplingPanel value={v} onChange={setV} defaults={null} footer={null} />
        <output data-testid="stop">{JSON.stringify(v.stop ?? [])}</output>
        <button type="button" onClick={() => setV({ values: {}, stop: ["from", "outside"] })}>
          swap
        </button>
      </>
    );
  }

  it("can be typed one character at a time, several of them, commas included", () => {
    render(<Stateful />);
    const input = screen.getByLabelText("Stop strings") as HTMLTextAreaElement;
    for (const ch of "foo,1\nbar") fireEvent.change(input, { target: { value: input.value + ch } });
    expect(input.value).toBe("foo,1\nbar");
    expect(screen.getByTestId("stop").textContent).toBe('["foo,1","bar"]');
  });

  it("follows a change from outside", () => {
    render(<Stateful />);
    fireEvent.click(screen.getByText("swap"));
    expect((screen.getByLabelText("Stop strings") as HTMLTextAreaElement).value).toBe("from\noutside");
  });
});

describe("usage history", () => {
  const panel = (v: DaemonUsageView) => <UsagePanel view={v} modelName="m" loadHash="h1" onLoadHash={() => undefined} includeConcurrent={false} onIncludeConcurrent={() => undefined} onClear={() => undefined} />;

  it("stays reachable when the current load has no requests yet", () => {
    render(panel(usage()));
    expect(screen.getByText("Load settings")).toBeTruthy();
    expect(screen.getByText(/No requests on these load settings yet/)).toBeTruthy();
    expect(screen.getByText("Clear history")).toBeTruthy();
    expect(screen.queryByText("Usage profile")).toBeNull();
  });

  it("says plainly when there is none at all", () => {
    render(panel({ ...usage(), loads: [] }));
    expect(screen.getByText(/Point an agent at the endpoint/)).toBeTruthy();
    expect(screen.queryByText("Load settings")).toBeNull();
  });
});

describe("the memory panel", () => {
  const memory = (over: Partial<DaemonMemorySnapshot>): DaemonMemorySnapshot => ({
    at: 1,
    unified: false,
    gpu_name: "RTX",
    gpu_count: 1,
    vram: { total: 8000, app: 1000, other: 1000, free: 6000 },
    ram: { total: 32000, app: 1000, other: 1000, free: 30000 },
    breakdown: { weights_gpu: null, kv_gpu: null, compute_gpu: null, weights_host: null, compute_host: null, prompt_cache: null, worker: 70 },
    spill: { measured: true, in_system_ram_mib: -10, spilled: false },
    ...over,
  });

  it("says when the figures may combine several GPUs", () => {
    render(<MemoryPanel status={status({ memory: memory({ gpu_count: 2 }) })} machineBusy={false} />);
    expect(screen.getByText(/has 2 GPUs/)).toBeTruthy();
  });

  it("says nothing about that with one", () => {
    render(<MemoryPanel status={status({ memory: memory({}) })} machineBusy={false} />);
    expect(screen.queryByText(/GPUs/)).toBeNull();
  });
});

describe("a key shown once", () => {
  it("is not carried to another machine", async () => {
    mocks.createDaemonKey.mockResolvedValue({ key: "lt-SECRETKEY", id: "k", name: "n", prefix: "lt-", created_at: 1, revoked_at: null });
    const props = { status: null, gateway: { host: "127.0.0.1", port: 8642, cors_origins: [], autostart: false }, keys: [], alias: null, onChanged: () => undefined };
    const { rerender, container } = render(<ConnectionPanel key="w1" workerId="w1" {...props} />);
    fireEvent.click(screen.getByText("Create key"));
    await waitFor(() => expect(container.textContent).toContain("lt-SECRETKEY"));
    rerender(<ConnectionPanel key="w2" workerId="w2" {...props} />);
    expect(container.textContent).not.toContain("lt-SECRETKEY");
  });
});
