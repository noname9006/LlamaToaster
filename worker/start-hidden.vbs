' Launches the LlamaToaster worker with no visible window.
' Copy this file into your Windows Startup folder (Win+R -> shell:startup)
' so it runs automatically at every login.
Set WshShell = CreateObject("WScript.Shell")

' Give Tailscale a moment to come up and assign this machine's tailnet IP
' before the worker tries to bind to it.
WScript.Sleep 5000

WshShell.CurrentDirectory = "F:\BOT\GitHub\LlamaToaster"
WshShell.Run "cmd /c npm run worker", 0, False
