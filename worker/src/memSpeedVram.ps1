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
$ErrorActionPreference = 'Stop'

function Emit-Failure([string]$reason) {
  [PSCustomObject]@{ ok = $false; reason = $reason } | ConvertTo-Json -Compress
  exit 0
}

try {
  $src = @'
using System; using System.Diagnostics; using System.Runtime.InteropServices; using System.Text;

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

  public static string DeviceName, FailReason;
  public static double GlobalMemMiB, ReadGBs, WriteGBs, CopyGBs;
  public static bool Ok;

  static void Chk(int e, string what) { if (e != 0) throw new Exception(what + " failed: OpenCL error " + e); }
  static string Info(IntPtr dev, uint p) {
    byte[] b = new byte[256]; UIntPtr r; clGetDeviceInfo(dev, p, (UIntPtr)256, b, out r);
    return Encoding.ASCII.GetString(b, 0, Math.Max(0, (int)r - 1));
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

  public static void Run(int mib) {
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

    foreach (string which in new string[] { "read", "write", "copy" }) {
      IntPtr k = which == "read" ? kRd : (which == "write" ? kWr : kCp);
      double factor = which == "copy" ? 2.0 : 1.0;
      int reps = 20; double best2 = 0;
      Chk(clEnqueueNDRangeKernel(q, k, 1, null, gsz, lsz, 0, null, IntPtr.Zero), "warm"); clFinish(q);
      for (int t = 0; t < 3; t++) {
        Stopwatch sw = Stopwatch.StartNew();
        for (int r = 0; r < reps; r++) Chk(clEnqueueNDRangeKernel(q, k, 1, null, gsz, lsz, 0, null, IntPtr.Zero), "run");
        clFinish(q); sw.Stop();
        double gb = (double)bytes * reps * factor / sw.Elapsed.TotalSeconds / 1e9;
        if (gb > best2) best2 = gb;
      }
      if (which == "read") ReadGBs = best2; else if (which == "write") WriteGBs = best2; else CopyGBs = best2;
    }
    Ok = true;
  }
}
'@
  $cp = New-Object System.CodeDom.Compiler.CompilerParameters
  $null = $cp.ReferencedAssemblies.Add('System.dll')
  Add-Type -TypeDefinition $src -Language CSharp -CompilerParameters $cp

  [Cl]::Run(256)
  if (-not [Cl]::Ok) { Emit-Failure([Cl]::FailReason) }
  [PSCustomObject]@{
    ok           = $true
    deviceName   = [Cl]::DeviceName
    globalMemMiB = [Cl]::GlobalMemMiB
    readGBs      = [Cl]::ReadGBs
    writeGBs     = [Cl]::WriteGBs
    copyGBs      = [Cl]::CopyGBs
  } | ConvertTo-Json -Compress
} catch {
  Emit-Failure($_.Exception.Message)
}
