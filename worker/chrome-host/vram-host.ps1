# LlamaToaster VRAM host (Chrome native messaging, Windows). Registered by the
# worker installer/updater (register.ps1) so the LlamaToaster Chrome extension can read VRAM.
# Reads every GPU through DXGI (Intel, AMD, NVIDIA, any vendor) and the Windows
# "GPU Adapter Memory" counters for live usage. Read-only, no dependencies.
$ErrorActionPreference = 'Stop'

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class LtDxgi {
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  public struct DESC1 {
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string Description;
    public uint VendorId, DeviceId, SubSysId, Revision;
    public UIntPtr DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory;
    public uint LuidLow; public int LuidHigh; public uint Flags;
  }
  [ComImport, Guid("29038f61-3839-4626-91fd-086879011a05"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IAdapter1 {
    void SetPrivateData(); void SetPrivateDataInterface(); void GetPrivateData(); void GetParent();
    void EnumOutputs(); void GetDesc(); void CheckInterfaceSupport();
    [PreserveSig] int GetDesc1(out DESC1 d);
  }
  [ComImport, Guid("770aae78-f26f-4dba-a829-253c83d1b387"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IFactory1 {
    void SetPrivateData(); void SetPrivateDataInterface(); void GetPrivateData(); void GetParent();
    void EnumAdapters(); void MakeWindowAssociation(); void GetWindowAssociation(); void CreateSwapChain(); void CreateSoftwareAdapter();
    [PreserveSig] int EnumAdapters1(uint i, out IAdapter1 a);
    [PreserveSig] bool IsCurrent();
  }
  [DllImport("dxgi.dll")] static extern int CreateDXGIFactory1(ref Guid riid, [MarshalAs(UnmanagedType.Interface)] out IFactory1 f);
  public static DESC1[] Adapters() {
    Guid g = typeof(IFactory1).GUID; IFactory1 f;
    Marshal.ThrowExceptionForHR(CreateDXGIFactory1(ref g, out f));
    var list = new System.Collections.Generic.List<DESC1>();
    for (uint i = 0; ; i++) {
      IAdapter1 a; if (f.EnumAdapters1(i, out a) != 0) break;
      DESC1 d; if (a.GetDesc1(out d) == 0 && (d.Flags & 2) == 0) list.Add(d);
      Marshal.ReleaseComObject(a);
    }
    Marshal.ReleaseComObject(f);
    return list.ToArray();
  }
}
'@

# Friendly app name for a process: the exe's FileDescription ("Google
# Chrome"), else the process name. Cached: it never changes for a name.
$script:AppNames = @{}
function App-Name($pr) {
  $n = $pr.ProcessName
  if (-not $script:AppNames.ContainsKey($n)) {
    $d = $null
    try { $d = $pr.MainModule.FileVersionInfo.FileDescription } catch {}
    $script:AppNames[$n] = $(if ($d -and $d.Trim()) { $d.Trim() } else { $n })
  }
  return $script:AppNames[$n]
}

# The apps holding the most RAM: private working set (what Task Manager's
# Memory column shows), summed over each app's processes. Above 256 MiB, top 8.
function Read-TopRam {
  $byPid = @{}
  foreach ($pr in Get-Process -ErrorAction SilentlyContinue) { $byPid[[int]$pr.Id] = $pr }
  $sum = @{}
  try {
    foreach ($c in Get-CimInstance Win32_PerfFormattedData_PerfProc_Process -Property IDProcess, WorkingSetPrivate) {
      if ([int]$c.IDProcess -eq 0) { continue }
      $pr = $byPid[[int]$c.IDProcess]
      if (-not $pr) { continue }
      $n = App-Name $pr
      $sum[$n] = [double]$sum[$n] + [double]$c.WorkingSetPrivate
    }
  } catch {
    # Perf counters disabled: the working set (includes shared pages).
    foreach ($pr in $byPid.Values) { $n = App-Name $pr; $sum[$n] = [double]$sum[$n] + [double]$pr.WorkingSet64 }
  }
  return @($sum.GetEnumerator() | Where-Object { $_.Value -ge 256MB } | Sort-Object Value -Descending | Select-Object -First 8 |
    ForEach-Object { [ordered]@{ name = $_.Key; mib = [math]::Round($_.Value / 1MB) } })
}

function Read-Gpus([bool]$wantApps = $false) {
  # Counter class has English property names on every Windows language.
  $usage = @{}
  try {
    foreach ($c in Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory) {
      if ($c.Name -match '^luid_0x([0-9a-fA-F]+)_0x([0-9a-fA-F]+)') {
        $usage[("{0}_{1}" -f $Matches[1], $Matches[2]).ToLower()] = $c
      }
    }
  } catch {}
  # What LlamaToaster's own llama.cpp processes hold, per adapter (LUID).
  $llamaPids = @{}
  foreach ($pr in Get-Process -ErrorAction SilentlyContinue) { if ($pr.ProcessName -like 'llama-*') { $llamaPids[[string]$pr.Id] = $true } }
  $script:LlamaRamMib = [math]::Round((($llamaPids.Keys | ForEach-Object { try { (Get-Process -Id $_ -ErrorAction Stop).WorkingSet64 } catch { 0 } } | Measure-Object -Sum).Sum) / 1MB)
  $llamaGpu = @{}
  # Per adapter: dedicated VRAM per app (only when the popup asks for apps).
  $appGpu = @{}
  if ($llamaPids.Count -gt 0 -or $wantApps) {
    $procs = @{}
    if ($wantApps) { foreach ($pr in Get-Process -ErrorAction SilentlyContinue) { $procs[[string]$pr.Id] = $pr } }
    try {
      foreach ($c in Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory) {
        if ($c.Name -notmatch '^pid_(\d+)_luid_0x([0-9a-fA-F]+)_0x([0-9a-fA-F]+)') { continue }
        $procId = $Matches[1]
        $k = ("{0}_{1}" -f $Matches[2], $Matches[3]).ToLower()
        if ($llamaPids[$procId]) {
          if (-not $llamaGpu[$k]) { $llamaGpu[$k] = @{ d = 0.0; s = 0.0 } }
          $llamaGpu[$k].d += [double]$c.DedicatedUsage; $llamaGpu[$k].s += [double]$c.SharedUsage
        }
        if ($wantApps -and $procs[$procId] -and [double]$c.DedicatedUsage -gt 0) {
          if (-not $appGpu[$k]) { $appGpu[$k] = @{} }
          $n = App-Name $procs[$procId]
          $appGpu[$k][$n] = [double]$appGpu[$k][$n] + [double]$c.DedicatedUsage
        }
      }
    } catch {}
  }
  $seen = @{}
  $out = @()
  foreach ($d in [LtDxgi]::Adapters()) {
    $key = ("{0:x8}_{1:x8}" -f $d.LuidHigh, $d.LuidLow)
    if ($seen[$key]) { continue }; $seen[$key] = $true
    $ded = [double]$d.DedicatedVideoMemory.ToUInt64()
    $shr = [double]$d.SharedSystemMemory.ToUInt64()
    $u = $usage[$key]
    $vendor = switch ($d.VendorId) { 0x10DE {'NVIDIA'} 0x1002 {'AMD'} 0x8086 {'Intel'} 0x5143 {'Qualcomm'} default { ('0x{0:X4}' -f $d.VendorId) } }
    # Integrated GPUs get a token carve-out (<= 512 MiB) and really run from shared RAM.
    $integrated = $ded -le 512MB
    $out += [ordered]@{
      name = $d.Description.Trim(); vendor = $vendor
      vendorId = [int]$d.VendorId; deviceId = [int]$d.DeviceId
      integrated = $integrated
      dedicatedTotalMib = [math]::Round($ded / 1MB)
      sharedTotalMib = [math]::Round($shr / 1MB)
      dedicatedUsedMib = $(if ($u) { [math]::Round([double]$u.DedicatedUsage / 1MB) } else { $null })
      sharedUsedMib = $(if ($u) { [math]::Round([double]$u.SharedUsage / 1MB) } else { $null })
      llamaDedicatedMib = $(if ($llamaGpu[$key]) { [math]::Round($llamaGpu[$key].d / 1MB) } else { 0 })
      llamaSharedMib = $(if ($llamaGpu[$key]) { [math]::Round($llamaGpu[$key].s / 1MB) } else { 0 })
      apps = $(if ($wantApps -and $appGpu[$key]) {
        @($appGpu[$key].GetEnumerator() | Where-Object { $_.Value -ge 128MB } | Sort-Object Value -Descending | Select-Object -First 8 |
          ForEach-Object { [ordered]@{ name = $_.Key; mib = [math]::Round($_.Value / 1MB) } })
      } else { $null })
    }
  }
  return $out
}

# Installed RAM from SMBIOS: type (SMBIOSMemoryType) and the speed it runs at.
function Read-Ram {
  $types = @{ 20 = 'DDR'; 21 = 'DDR2'; 24 = 'DDR3'; 26 = 'DDR4'; 27 = 'LPDDR'; 28 = 'LPDDR2'; 29 = 'LPDDR3'; 30 = 'LPDDR4'; 34 = 'DDR5'; 35 = 'LPDDR5' }
  $mods = @(Get-CimInstance Win32_PhysicalMemory -ErrorAction Stop)
  if ($mods.Count -eq 0) { return $null }
  $type = ($mods | ForEach-Object { $types[[int]$_.SMBIOSMemoryType] } | Where-Object { $_ } | Group-Object | Sort-Object Count -Descending | Select-Object -First 1).Name
  $run = @($mods | ForEach-Object { if ($_.ConfiguredClockSpeed) { [int]$_.ConfiguredClockSpeed } elseif ($_.Speed) { [int]$_.Speed } } | Where-Object { $_ -gt 0 })
  $rated = @($mods | ForEach-Object { [int]$_.Speed } | Where-Object { $_ -gt 0 })
  $mts = $(if ($run.Count) { ($run | Measure-Object -Minimum).Minimum } else { $null })
  $ratedMin = $(if ($rated.Count) { ($rated | Measure-Object -Minimum).Minimum } else { $null })
  return [ordered]@{ type = $type; mts = $mts; ratedMts = $(if ($ratedMin -and $mts -and $ratedMin -gt $mts) { $ratedMin } else { $null }); modules = $mods.Count }
}

$stdin = [Console]::OpenStandardInput()
$stdout = [Console]::OpenStandardOutput()
function Send($obj) {
  $b = [Text.Encoding]::UTF8.GetBytes(($obj | ConvertTo-Json -Depth 5 -Compress))
  $stdout.Write([BitConverter]::GetBytes([int]$b.Length), 0, 4); $stdout.Write($b, 0, $b.Length); $stdout.Flush()
}
if ($args -contains '--print') { ConvertTo-Json -InputObject @(Read-Gpus $true) -Depth 5; "llamaRamMib=$($script:LlamaRamMib)"; ConvertTo-Json -InputObject (Read-Ram) -Compress; ConvertTo-Json -InputObject @(Read-TopRam) -Compress; exit 0 }
$len = New-Object byte[] 4
while ($stdin.Read($len, 0, 4) -eq 4) {
  $n = [BitConverter]::ToInt32($len, 0); $buf = New-Object byte[] $n; $got = 0
  while ($got -lt $n) { $r = $stdin.Read($buf, $got, $n - $got); if ($r -le 0) { exit 0 }; $got += $r }
  $wantRam = $false; $wantApps = $false
  try { $req = [Text.Encoding]::UTF8.GetString($buf) | ConvertFrom-Json; $wantRam = [bool]$req.ram; $wantApps = [bool]$req.apps } catch {}
  try {
    $g = @(Read-Gpus $wantApps)
    $ram = $null
    if ($wantRam) { try { $ram = Read-Ram } catch {} }
    $topRam = $null
    if ($wantApps) { try { $topRam = @(Read-TopRam) } catch {} }
    Send @{ ok = $true; helper = 1; os = 'windows'; gpus = $g; ram = $ram; llamaRamMib = $script:LlamaRamMib; topRam = $topRam }
  }
  catch { Send @{ ok = $false; helper = 1; error = $_.Exception.Message } }
}
