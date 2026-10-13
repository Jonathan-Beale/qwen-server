# Installs the Qwen agent on a Windows machine and puts a "Qwen" icon on the desktop.
# The model keeps running on the server; this machine connects to it over SSH.
#
#   scp user@server:qwen-server/client/install.ps1 $env:TEMP\install.ps1
#   powershell -ExecutionPolicy Bypass -File $env:TEMP\install.ps1 -Server user@server
#
# Re-run it at any time to update to the server's current version.

param(
    [Parameter(Mandatory = $true)][string]$Server,   # e.g. jon@overlord
    [string]$RemoteDir = "qwen-server",              # repo location on the server, relative to its home
    [string]$Workspace = (Join-Path $HOME "qwen-workspace")
)

$InstallDir = Join-Path $env:LOCALAPPDATA "qwen-agent"

function Fail($message) {
    Write-Host "ERROR: $message" -ForegroundColor Red
    exit 1
}

# 1. Python 3.10+
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Fail "Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run this again."
}
$pythonExe = & python -c "import sys; print(sys.executable if sys.version_info >= (3, 10) else '')"
if (-not $pythonExe) {
    Fail "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/ and run this again."
}
$launcher = Join-Path (Split-Path $pythonExe) "pythonw.exe"   # pythonw runs without a console window
if (-not (Test-Path $launcher)) { $launcher = $pythonExe }

# 2. Passwordless SSH to the server (the desktop icon cannot type a password)
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Fail "The OpenSSH client was not found. Add it under Settings > System > Optional features, then run this again."
}
cmd /c "ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new $Server exit >nul 2>&1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Setting up an SSH key for $Server ..."
    $key = Join-Path $HOME ".ssh\id_ed25519"
    if (-not (Test-Path "$key.pub")) {
        New-Item -ItemType Directory -Force (Join-Path $HOME ".ssh") | Out-Null
        Write-Host "Creating a key. Press Enter twice when asked for a passphrase (leave it empty)."
        ssh-keygen -t ed25519 -f $key
    }
    Write-Host "Enter your password for $Server one last time:"
    Get-Content "$key.pub" | ssh $Server "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    cmd /c "ssh -o BatchMode=yes -o ConnectTimeout=10 $Server exit >nul 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Fail "Could not set up passwordless SSH to $Server. Check that 'ssh $Server' works, then run this again."
    }
}

# 3. Copy the agent from the server
New-Item -ItemType Directory -Force $InstallDir | Out-Null
New-Item -ItemType Directory -Force $Workspace | Out-Null
scp "${Server}:$RemoteDir/scripts/agent.py" "${Server}:$RemoteDir/scripts/agent_ui.html" $InstallDir
if ($LASTEXITCODE -ne 0) {
    Fail "Could not copy the agent from ${Server}:$RemoteDir. Pass -RemoteDir if the repo is somewhere else."
}

# 4. Desktop shortcut
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $desktop "Qwen.lnk"))
$shortcut.TargetPath = $launcher
$shortcut.Arguments = '"{0}" --ui --ssh {1}' -f (Join-Path $InstallDir "agent.py"), $Server
$shortcut.WorkingDirectory = $Workspace
$shortcut.Description = "Chat with the Qwen model on $Server"
$shortcut.Save()

Write-Host ""
Write-Host "Installed. Double-click 'Qwen' on your desktop to start." -ForegroundColor Green
Write-Host "The assistant works in $Workspace until you pick another folder with 'Change folder'."
