' Launches the LlamaToaster worker with no visible window.
' Create a SHORTCUT to this file (not a copy -- it locates the repo via its
' own path, which only works if it keeps running from here) in your Windows
' Startup folder (Win+R -> shell:startup) so it runs automatically at every
' login.
Set WshShell = CreateObject("WScript.Shell")

' Give Tailscale a moment to come up and assign this machine's tailnet IP
' before the worker tries to bind to it.
WScript.Sleep 5000

Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
repoRoot = fso.GetParentFolderName(scriptDir)

WshShell.CurrentDirectory = repoRoot
WshShell.Run "cmd /c npm run worker", 0, False
