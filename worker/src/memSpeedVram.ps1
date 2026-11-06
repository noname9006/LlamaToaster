# Windows-only VRAM read/write/copy bandwidth measurement, driven by
# worker/src/memSpeed.ts's measureVramBandwidth. Picks the GPU device with
# the most global memory across every OpenCL platform (works for NVIDIA,
# AMD, and Intel alike -- unlike nvidia-smi, which only covers NVIDIA), runs
# a short read/write/copy sweep, and prints exactly one line of compressed
# JSON to stdout: {"ok":true,"deviceName":...,"globalMemMiB":...,
# "readGBs":...,"writeGBs":...,"copyGBs":...} on success, or
# {"ok":false,"reason":"..."} otherwise. Always exits 0 -- errors are
# reported IN the JSON, not via exit code, so the worker only has to parse
# stdout.
#
# Quick mode (default): one 256 MiB pair of buffers; each cell is the MEDIAN
# of five timed windows of ~150 ms (the repetition count is calibrated from a
# warm-up run), not the single luckiest series.
# -Deep: on top of that, allocates as many 256 MiB chunks as fit in the card's
# memory minus a safety margin (the larger of 768 MiB and 10%: the desktop and
# other apps already hold some), commits each one, then measures read/write/
# copy on every chunk (best of three windows per chunk). The per-chunk arrays
# come back in the JSON under "deep" and worker/src/memSpeed.ts summarises them
# (median/min/max/slow chunks). A slow tail usually means the driver began
# spilling into shared system memory, i.e. the real on-card VRAM ran out.
param([switch]$Deep)
$ErrorActionPreference = 'Stop'

function Emit-Failure([string]$reason) {
  [PSCustomObject]@{ ok = $false; reason = $reason } | ConvertTo-Json -Compress
  exit 0
}

try {
  $src = @'
using System; using System.Collections.Generic; using System.Diagnostics; using System.Runtime.InteropServices; using System.Text;

public static class Cl {
  const string D = "OpenCL.dll";
  [DllImport(D)] static extern int clGetPlatformIDs(uint n, IntPtr[] p, out uint np);
  [DllImport(D)] static extern int clGetDeviceIDs(IntPtr plat, ulong type, uint n, IntPtr[] d, out uint nd);
  [DllImport(D)] static extern int clGetDeviceInfo(IntPtr dev, uint param, UIntPtr sz, byte[] val, out UIntPtr ret);
  [DllImport(D)] static extern IntPtr clCreateContext(IntPtr props, uint n, IntPtr[] devs, IntPtr cb, IntPtr ud, out int err);
  [DllImport(D)] static extern IntPtr clCreateCommandQueue(IntPtr ctx, IntPtr dev, ulong props, out int err);
  [DllImport(D)] static extern IntPtr clCreateBuffer(IntPtr ctx, ulong flags, UIntPtr size, IntPtr host, out int err);
  [DllImport(D)] static extern IntPtr clCreateProgramWithSource(IntPtr ctx, uint count, string[] src, UIntPtr[] lens, out int err);
  [DllImport(D)] static extern int clBuildProgram(IntPtr prog, uint n, IntPtr[] devs, string opts, IntPtr cb, IntPtr ud);
  [DllImport(D)] static extern int clGetProgramBuildInfo(IntPtr prog, IntPtr dev, uint param, UIntPtr sz, byte[] val, out UIntPtr ret);
  [DllImport(D)] static extern IntPtr clCreateKernel(IntPtr prog, string name, out int err);
  [DllImport(D)] static extern int clSetKernelArg(IntPtr k, uint idx, UIntPtr sz, ref IntPtr val);
  [DllImport(D)] static extern int clSetKernelArg(IntPtr k, uint idx, UIntPtr sz, ref uint val);
  [DllImport(D)] static extern int clEnqueueNDRangeKernel(IntPtr q, IntPtr k, uint dim, UIntPtr[] off, UIntPtr[] gsz, UIntPtr[] lsz, uint nev, IntPtr[] evs, IntPtr ev);
  [DllImport(D)] static extern int clEnqueueWriteBuffer(IntPtr q, IntPtr buf, uint block, UIntPtr off, UIntPtr size, IntPtr ptr, uint nev, IntPtr[] evs, IntPtr ev);
  [DllImport(D)] static extern int clFinish(IntPtr q);

  public static string DeviceName, FailReason, DeepSkipped;
  public static double GlobalMemMiB, ReadGBs, WriteGBs, CopyGBs;
  public static int ChunkMiB;
  public static double[] ReadChunks, WriteChunks, CopyChunks;
  public static bool Ok;

  static void Chk(int e, string what) { if (e != 0) throw new Exception(what + " failed: OpenCL error " + e); }
  static string Info(IntPtr dev, uint p) {
    byte[] b = new byte[256]; UIntPtr r; clGetDeviceInfo(dev, p, (UIntPtr)256, b, out r);
    return Encoding.ASCII.GetString(b, 0, Math.Max(0, (int)r - 1));
  }
  static void SetBuf(IntPtr k, uint idx, IntPtr buf) { Chk(clSetKernelArg(k, idx, (UIntPtr)IntPtr.Size, ref buf), "set kernel arg"); }

  // Seconds taken by `reps` back-to-back launches of kernel k.
  static double Window(IntPtr q, IntPtr k, UIntPtr[] gsz, UIntPtr[] lsz, int reps) {
    Stopwatch sw = Stopwatch.StartNew();
    for (int r = 0; r < reps; r++) Chk(clEnqueueNDRangeKernel(q, k, 1, null, gsz, lsz, 0, null, IntPtr.Zero), "run");
    Chk(clFinish(q), "finish");
    sw.Stop();
    return sw.Elapsed.TotalSeconds;
  }

  static double Median(double[] v) {
    double[] s = (double[])v.Clone(); Array.Sort(s); int n = s.Length;
    return (n % 2 == 1) ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2.0;
  }

  // Quick-mode cell: two warm-ups (the second calibrates the repetition count
  // for a ~150 ms window), then five windows; the median GB/s.
  static double Quick(IntPtr q, IntPtr k, UIntPtr[] gsz, UIntPtr[] lsz, ulong bytes, double factor) {
    Window(q, k, gsz, lsz, 1);
    double one = Window(q, k, gsz, lsz, 1);
    int reps = (int)Math.Ceiling(0.15 / Math.Max(one, 1e-6));
    if (reps < 1) reps = 1; if (reps > 400) reps = 400;
    double[] s = new double[5];
    for (int t = 0; t < 5; t++) s[t] = (double)bytes * reps * factor / Window(q, k, gsz, lsz, reps) / 1e9;
    return Median(s);
  }

  // Deep-mode cell: one warm-up, then the best of three 20-launch windows.
  static double Best(IntPtr q, IntPtr k, UIntPtr[] gsz, UIntPtr[] lsz, ulong bytes, double factor) {
    Window(q, k, gsz, lsz, 1);
    double best = 0;
    for (int t = 0; t < 3; t++) {
      double gb = (double)bytes * 20 * factor / Window(q, k, gsz, lsz, 20) / 1e9;
      if (gb > best) best = gb;
    }
    return best;
  }

  static ulong InfoU(IntPtr dev, uint p) { byte[] b = new byte[8]; UIntPtr r; clGetDeviceInfo(dev, p, (UIntPtr)8, b, out r); return BitConverter.ToUInt64(b, 0); }

  const string KSRC = @"
__kernel void rd(__global const float4* a, __global float* out, uint n) {
  size_t gid = get_global_id(0), gs = get_global_size(0); float4 acc = (float4)(0.0f);
  for (size_t i = gid; i < n; i += gs) acc += a[i];
  if (acc.x + acc.y + acc.z + acc.w == 123456.0f) out[0] = 1.0f;
}
__kernel void wr(__global float4* b, uint n) {
  size_t gid = get_global_id(0), gs = get_global_size(0);
  for (size_t i = gid; i < n; i += gs) b[i] = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
}
__kernel void cp(__global const float4* a, __global float4* b, uint n) {
  size_t gid = get_global_id(0), gs = get_global_size(0);
  for (size_t i = gid; i < n; i += gs) b[i] = a[i];
}";

  // Allocates chunk-sized buffers until the card's memory (minus a margin) is
  // used or the driver refuses, commits each by writing it once (some drivers
  // allocate lazily), then measures every chunk. The copy kernel's second
  // argument (buffer B, the copy destination) is already bound and stays put.
  // The two quick-mode buffers (A and B, 2 chunks' worth) are still alive, so
  // they are counted against the budget.
  static void DeepScan(IntPtr ctx, IntPtr q, IntPtr kRd, IntPtr kWr, IntPtr kCp,
                       UIntPtr[] gsz, UIntPtr[] lsz, ulong chunkBytes, ulong globalBytes, int mib) {
    ulong reserve = Math.Max(768UL * 1024 * 1024, globalBytes / 10);
    if (globalBytes <= reserve + 4 * chunkBytes) { DeepSkipped = "not enough VRAM for a deep scan"; return; }
    int maxChunks = (int)Math.Min(96UL, (globalBytes - reserve - 2 * chunkBytes) / chunkBytes);
    List<IntPtr> chunks = new List<IntPtr>();
    for (int i = 0; i < maxChunks; i++) {
      int err; IntPtr c = clCreateBuffer(ctx, 1, (UIntPtr)chunkBytes, IntPtr.Zero, out err);
      if (err != 0 || c == IntPtr.Zero) break;
      SetBuf(kWr, 0, c);
      if (clEnqueueNDRangeKernel(q, kWr, 1, null, gsz, lsz, 0, null, IntPtr.Zero) != 0) break;
      if (clFinish(q) != 0) break;
      chunks.Add(c);
    }
    if (chunks.Count < 2) { DeepSkipped = "could not allocate enough VRAM for a deep scan"; return; }
    double[] rd = new double[chunks.Count], wr = new double[chunks.Count], cp = new double[chunks.Count];
    for (int i = 0; i < chunks.Count; i++) {
      SetBuf(kRd, 0, chunks[i]); SetBuf(kWr, 0, chunks[i]); SetBuf(kCp, 0, chunks[i]);
      rd[i] = Best(q, kRd, gsz, lsz, chunkBytes, 1.0);
      wr[i] = Best(q, kWr, gsz, lsz, chunkBytes, 1.0);
      cp[i] = Best(q, kCp, gsz, lsz, chunkBytes, 2.0);
    }
    ChunkMiB = mib; ReadChunks = rd; WriteChunks = wr; CopyChunks = cp;
  }

  public static void Run(int mib, bool deep) {
    uint np; Chk(clGetPlatformIDs(0, null, out np), "platforms");
    if (np == 0) { Ok = false; FailReason = "no OpenCL platform found"; return; }
    IntPtr[] plats = new IntPtr[np]; clGetPlatformIDs(np, plats, out np);
    IntPtr best = IntPtr.Zero; ulong bestMem = 0;
    foreach (IntPtr p in plats) {
      uint nd; if (clGetDeviceIDs(p, 4, 0, null, out nd) != 0 || nd == 0) continue;
      IntPtr[] ds = new IntPtr[nd]; clGetDeviceIDs(p, 4, nd, ds, out nd);
      foreach (IntPtr d in ds) { ulong m = InfoU(d, 0x101F); if (m > bestMem) { bestMem = m; best = d; } }
    }
    if (best == IntPtr.Zero) { Ok = false; FailReason = "no OpenCL GPU device found"; return; }
    DeviceName = Info(best, 0x102B); GlobalMemMiB = bestMem / 1024.0 / 1024.0;

    int err;
    IntPtr[] devs = new IntPtr[] { best };
    IntPtr ctx = clCreateContext(IntPtr.Zero, 1, devs, IntPtr.Zero, IntPtr.Zero, out err); Chk(err, "context");
    IntPtr q = clCreateCommandQueue(ctx, best, 0, out err); Chk(err, "queue");
    IntPtr prog = clCreateProgramWithSource(ctx, 1, new string[] { KSRC }, null, out err); Chk(err, "program");
    int be = clBuildProgram(prog, 1, devs, "", IntPtr.Zero, IntPtr.Zero);
    if (be != 0) {
      byte[] log = new byte[8192]; UIntPtr r; clGetProgramBuildInfo(prog, best, 0x1183, (UIntPtr)8192, log, out r);
      Ok = false; FailReason = "OpenCL build failed: " + Encoding.ASCII.GetString(log, 0, Math.Max(0, (int)r)); return;
    }
    IntPtr kRd = clCreateKernel(prog, "rd", out err); Chk(err, "kernel rd");
    IntPtr kWr = clCreateKernel(prog, "wr", out err); Chk(err, "kernel wr");
    IntPtr kCp = clCreateKernel(prog, "cp", out err); Chk(err, "kernel cp");

    ulong bytes = (ulong)mib * 1024 * 1024; uint n = (uint)(bytes / 16);
    IntPtr A = clCreateBuffer(ctx, 1, (UIntPtr)bytes, IntPtr.Zero, out err); Chk(err, "alloc A");
    IntPtr B = clCreateBuffer(ctx, 1, (UIntPtr)bytes, IntPtr.Zero, out err); Chk(err, "alloc B");
    IntPtr O = clCreateBuffer(ctx, 1, (UIntPtr)16, IntPtr.Zero, out err); Chk(err, "alloc O");
    IntPtr host = Marshal.AllocHGlobal((int)(32 * 1024 * 1024));
    for (ulong off = 0; off < bytes; off += 32UL * 1024 * 1024)
      Chk(clEnqueueWriteBuffer(q, A, 1, (UIntPtr)off, (UIntPtr)Math.Min(32UL * 1024 * 1024, bytes - off), host, 0, null, IntPtr.Zero), "fill A");
    Marshal.FreeHGlobal(host);

    IntPtr a = A, b = B, o = O; uint nn = n;
    clSetKernelArg(kRd, 0, (UIntPtr)IntPtr.Size, ref a); clSetKernelArg(kRd, 1, (UIntPtr)IntPtr.Size, ref o); clSetKernelArg(kRd, 2, (UIntPtr)4, ref nn);
    clSetKernelArg(kWr, 0, (UIntPtr)IntPtr.Size, ref b); clSetKernelArg(kWr, 1, (UIntPtr)4, ref nn);
    clSetKernelArg(kCp, 0, (UIntPtr)IntPtr.Size, ref a); clSetKernelArg(kCp, 1, (UIntPtr)IntPtr.Size, ref b); clSetKernelArg(kCp, 2, (UIntPtr)4, ref nn);
    UIntPtr[] gsz = new UIntPtr[] { (UIntPtr)(1 << 20) }; UIntPtr[] lsz = new UIntPtr[] { (UIntPtr)256 };

    ReadGBs = Quick(q, kRd, gsz, lsz, bytes, 1.0);
    WriteGBs = Quick(q, kWr, gsz, lsz, bytes, 1.0);
    CopyGBs = Quick(q, kCp, gsz, lsz, bytes, 2.0);

    if (deep) DeepScan(ctx, q, kRd, kWr, kCp, gsz, lsz, bytes, bestMem, mib);
    Ok = true;
  }
}
'@
  $cp = New-Object System.CodeDom.Compiler.CompilerParameters
  $null = $cp.ReferencedAssemblies.Add('System.dll')
  Add-Type -TypeDefinition $src -Language CSharp -CompilerParameters $cp

  [Cl]::Run(256, [bool]$Deep)
  if (-not [Cl]::Ok) { Emit-Failure([Cl]::FailReason) }
  $out = [ordered]@{
    ok           = $true
    deviceName   = [Cl]::DeviceName
    globalMemMiB = [Cl]::GlobalMemMiB
    readGBs      = [Cl]::ReadGBs
    writeGBs     = [Cl]::WriteGBs
    copyGBs      = [Cl]::CopyGBs
  }
  if ($null -ne [Cl]::ReadChunks) {
    $out.deep = [ordered]@{
      chunkMiB    = [Cl]::ChunkMiB
      readChunks  = @([Cl]::ReadChunks)
      writeChunks = @([Cl]::WriteChunks)
      copyChunks  = @([Cl]::CopyChunks)
    }
  }
  if ([Cl]::DeepSkipped) { $out.deepSkippedReason = [Cl]::DeepSkipped }
  $out | ConvertTo-Json -Compress -Depth 4
} catch {
  Emit-Failure($_.Exception.Message)
}
