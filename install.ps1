# Promptwood on Windows: one command from nothing to a working app (the counterpart of install.sh).
#
#   powershell -ExecutionPolicy Bypass -File install.ps1                 asks what to install, shows the plan, installs
#   powershell -ExecutionPolicy Bypass -File install.ps1 --profile core  video clips only
#   powershell -ExecutionPolicy Bypass -File install.ps1 --dry-run       only show what would be installed
#   powershell -ExecutionPolicy Bypass -File install.ps1 --help          every option
#
# STATUS: prepared, not yet run on a Windows computer. Linux is the tested system (install.sh).
#
# It needs no administrator rights and writes into two folders only: this one (the app and its .venv) and the home
# folder (%VIDEO_CHAIN_HOME%, default %LOCALAPPDATA%\video-chain: ComfyUI, models, tools).
$ErrorActionPreference = "Stop"
function Die($msg) { Write-Host ""; Write-Host "Install stopped: $msg" -ForegroundColor Red; exit 1 }

$HomeDir = if ($env:VIDEO_CHAIN_HOME) { $env:VIDEO_CHAIN_HOME } else { Join-Path $env:LOCALAPPDATA "video-chain" }
$env:VIDEO_CHAIN_HOME = $HomeDir
$env:PYTHONUTF8 = "1"                       # text files as UTF-8, as on Linux
$ToolsBin = Join-Path $HomeDir "tools\bin"
New-Item -ItemType Directory -Force -Path $ToolsBin | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $HomeDir "logs") | Out-Null

# ---- the app's source: this folder
$Src = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $Src "pyproject.toml"))) { Die "run install.ps1 from Promptwood's folder" }

# ---- uv (makes the Python environments and brings Python 3.12 itself when the system has none)
$Uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $Uv -and (Test-Path (Join-Path $ToolsBin "uv.exe"))) { $Uv = Join-Path $ToolsBin "uv.exe" }
if (-not $Uv) {
    Write-Host "Getting uv, the tool that sets up Python environments (into $ToolsBin; nothing else on the system is changed) ..."
    $env:UV_UNMANAGED_INSTALL = $ToolsBin
    try { Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression | Out-Null }
    catch { Die "could not install uv (https://docs.astral.sh/uv/getting-started/installation/): $_" }
    $Uv = Join-Path $ToolsBin "uv.exe"
    if (-not (Test-Path $Uv)) { Die "uv was not installed to $Uv" }
}
$env:PATH = (Split-Path -Parent $Uv) + ";" + $env:PATH
if (-not $env:UV_HTTP_TIMEOUT) { $env:UV_HTTP_TIMEOUT = "300" }

# ---- first the questions, the plan and the checks: before anything big is installed
function Ask {
    Push-Location $Src
    try {
        $env:PYTHONPATH = Join-Path $Src "src"
        & $Uv run -q --no-project --python 3.12 --with pyyaml python -X utf8 -m video_chain.setup setup --phase ask @args
        return $LASTEXITCODE
    } finally { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue; Pop-Location }
}
if ($args -contains "--help" -or $args -contains "-h" -or $args -contains "--dry-run") { exit (Ask @args) }
Write-Host "Preparing (uv fetches a small Python for the installer the first time) ..."
$code = Ask @args
if ($code -ne 0) { exit $code }

# ---- the app in its own environment (the versions the app was tested with: uv.lock)
$AppLog = Join-Path $HomeDir "logs\app.log"
Write-Host ""
Write-Host "Installing Promptwood itself into $Src\.venv (about 1 minute; details in $AppLog) ..."
$ok = $false
foreach ($try in 1..3) {
    Push-Location $Src
    try { & $Uv sync --locked --extra faces --python 3.12 *>> $AppLog; if ($LASTEXITCODE -eq 0) { $ok = $true } }
    finally { Pop-Location }
    if ($ok) { break }
    Write-Host "A download did not get through; trying again ..."
    Start-Sleep -Seconds (10 * $try)
}
if (-not $ok) { Get-Content $AppLog -Tail 25; Die "could not install the app's Python packages (all of it is in $AppLog)" }

# ---- everything else: ComfyUI, models, text to speech, llama.cpp, ffmpeg; then the checks and the tests
& (Join-Path $Src ".venv\Scripts\python.exe") -X utf8 -m video_chain setup --answers @args
exit $LASTEXITCODE
