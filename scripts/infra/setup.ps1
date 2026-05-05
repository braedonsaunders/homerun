# Homerun - Windows Setup Script
# Run: .\scripts\infra\setup.ps1

param(
    [switch]$PostgresOnly,
    [switch]$NoBanner
)

$ErrorActionPreference = "Stop"

# Navigate to project root (grandparent of scripts\infra\)
Set-Location (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)))

# ── Banner & UI Helpers ──────────────────────────────────────────────

function Show-Banner {
    Write-Host ""
    Write-Host "    $([char]27)[38;2;30;107;69m██   ██  ██████  ███    ███ ███████ ██████  ██    ██ ███    ██$([char]27)[0m"
    Write-Host "    $([char]27)[38;2;35;138;85m██   ██ ██    ██ ████  ████ ██      ██   ██ ██    ██ ████   ██$([char]27)[0m"
    Write-Host "    $([char]27)[38;2;42;172;104m███████ ██    ██ ██ ████ ██ █████   ██████  ██    ██ ██ ██  ██$([char]27)[0m"
    Write-Host "    $([char]27)[38;2;53;212;122m██   ██ ██    ██ ██  ██  ██ ██      ██   ██ ██    ██ ██  ██ ██$([char]27)[0m"
    Write-Host "    $([char]27)[38;2;88;241;193m██   ██  ██████  ██      ██ ███████ ██   ██  ██████  ██   ████$([char]27)[0m"
    Write-Host ""
    Write-Host "                    Autonomous Trading Platform" -ForegroundColor DarkGray
    Write-Host ""
}

$script:stepNum = 0
$script:totalSteps = 8

function Show-Step {
    param([string]$Text)
    $script:stepNum++
    $num = "$($script:stepNum)/$($script:totalSteps)"
    $pad = 46 - $Text.Length
    if ($pad -lt 2) { $pad = 2 }
    $dots = " " + ("." * $pad) + " "
    Write-Host -NoNewline "    [$num]  $Text$dots"
}

function Show-StepOK {
    param([string]$Detail = "")
    if ($Detail) {
        Write-Host "OK  $Detail" -ForegroundColor Green
    } else {
        Write-Host "OK" -ForegroundColor Green
    }
}

function Show-StepWarn {
    param([string]$Detail = "")
    if ($Detail) {
        Write-Host "!!  $Detail" -ForegroundColor Yellow
    } else {
        Write-Host "!!" -ForegroundColor Yellow
    }
}

function Show-StepFail {
    param([string]$Detail = "")
    if ($Detail) {
        Write-Host "FAIL  $Detail" -ForegroundColor Red
    } else {
        Write-Host "FAIL" -ForegroundColor Red
    }
}

function Show-SubInfo {
    param([string]$Text)
    Write-Host "             $Text" -ForegroundColor DarkGray
}

# ── Utility Functions ────────────────────────────────────────────────

function Get-InstallerLogDir {
    $logDir = Join-Path (Get-Location).Path "data\runtime\logs"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    return $logDir
}

function Show-LogTail {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path $Path)) {
        return
    }

    Write-Host "$Label (tail):" -ForegroundColor Yellow
    Get-Content -Path $Path -Tail 30 -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  $_" -ForegroundColor DarkYellow
    }
}

function Find-PostgresBinDir {
    if ($env:POSTGRES_BIN_DIR) {
        $envBin = $env:POSTGRES_BIN_DIR
        if (
            (Test-Path (Join-Path $envBin "initdb.exe")) -and
            (Test-Path (Join-Path $envBin "pg_ctl.exe"))
        ) {
            return $envBin
        }
    }

    $initdb = Get-Command initdb -ErrorAction SilentlyContinue
    $pgctl = Get-Command pg_ctl -ErrorAction SilentlyContinue
    if ($initdb -and $pgctl) {
        return (Split-Path -Parent $initdb.Source)
    }

    $roots = @(
        "C:\Program Files\PostgreSQL",
        "C:\Program Files (x86)\PostgreSQL"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) {
            continue
        }
        $versions = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        foreach ($versionDir in $versions) {
            $bin = Join-Path $versionDir.FullName "bin"
            if (
                (Test-Path (Join-Path $bin "initdb.exe")) -and
                (Test-Path (Join-Path $bin "pg_ctl.exe"))
            ) {
                return $bin
            }
        }
    }

    return $null
}

function Get-DockerCliPath {
    $cmd = Get-Command docker -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        (Join-Path $env:ProgramFiles "Docker\Docker\resources\bin\docker.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Docker\Docker\resources\bin\docker.exe"),
        (Join-Path $env:LocalAppData "Programs\Docker\Docker\resources\bin\docker.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    return $null
}

function Ensure-DockerCommand {
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        return $true
    }

    $dockerPath = Get-DockerCliPath
    if (-not $dockerPath) {
        return $false
    }

    try {
        Set-Alias -Name docker -Value $dockerPath -Scope Script -Force
        return $true
    } catch {
        return $false
    }
}

function Test-DockerRuntimeAvailable {
    if (-not (Ensure-DockerCommand)) {
        return $false
    }

    try {
        docker info *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Get-DockerDesktopExePath {
    $candidates = @(
        "C:\Program Files\Docker\Docker\Docker Desktop.exe",
        "C:\Program Files\Docker\Docker\Docker Desktop Installer.exe",
        (Join-Path $env:LocalAppData "Programs\Docker\Docker\Docker Desktop.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }
    return $null
}

function Wait-ForDockerRuntime {
    param([int]$TimeoutSeconds = 240)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $desktopLaunched = $false
    while ((Get-Date) -lt $deadline) {
        if (Test-DockerRuntimeAvailable) {
            return $true
        }
        try {
            Start-Service -Name "com.docker.service" -ErrorAction SilentlyContinue
        } catch {
        }
        if (-not $desktopLaunched) {
            $desktopExe = Get-DockerDesktopExePath
            if ($desktopExe) {
                try {
                    Start-Process -FilePath $desktopExe -WindowStyle Hidden | Out-Null
                    $desktopLaunched = $true
                } catch {
                }
            }
        }
        Start-Sleep -Seconds 3
    }
    return (Test-DockerRuntimeAvailable)
}

function Try-InstallDockerWithWinget {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        return $false
    }

    $logDir = Get-InstallerLogDir
    $logPath = Join-Path $logDir "docker-winget-install.log"
    Show-SubInfo "Installing Docker Desktop via winget..."
    try {
        & winget install --id Docker.DockerDesktop --exact --silent --disable-interactivity --accept-source-agreements --accept-package-agreements 2>&1 | Tee-Object -FilePath $logPath | Out-Null
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0 -and $exitCode -ne 3010) {
            Show-SubInfo "Winget Docker install failed (exit code $exitCode). Log: $logPath"
            Show-LogTail -Path $logPath -Label "winget output"
            return $false
        }
        if (Wait-ForDockerRuntime -TimeoutSeconds 300) {
            return $true
        }
        Show-SubInfo "Docker install completed but runtime not ready. Log: $logPath"
        Show-LogTail -Path $logPath -Label "winget output"
        return $false
    } catch {
        Show-SubInfo "Winget Docker install threw: $($_.Exception.Message)"
        return $false
    }
}

function Try-InstallDockerWithChocolatey {
    $choco = Get-Command choco -ErrorAction SilentlyContinue
    if (-not $choco) {
        return $false
    }

    $logDir = Get-InstallerLogDir
    $logPath = Join-Path $logDir "docker-choco-install.log"
    Show-SubInfo "Installing Docker Desktop via Chocolatey..."
    try {
        & choco install docker-desktop -y --no-progress 2>&1 | Tee-Object -FilePath $logPath | Out-Null
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            Show-SubInfo "Chocolatey Docker install failed (exit code $exitCode). Log: $logPath"
            Show-LogTail -Path $logPath -Label "choco output"
            return $false
        }
        if (Wait-ForDockerRuntime -TimeoutSeconds 300) {
            return $true
        }
        Show-SubInfo "Docker install completed but runtime not ready. Log: $logPath"
        Show-LogTail -Path $logPath -Label "choco output"
        return $false
    } catch {
        Show-SubInfo "Chocolatey Docker install threw: $($_.Exception.Message)"
        return $false
    }
}

function Ensure-DockerRuntime {
    if (Test-DockerRuntimeAvailable) {
        return $true
    }
    if (Wait-ForDockerRuntime -TimeoutSeconds 45) {
        return $true
    }

    Show-SubInfo "Docker runtime missing. Attempting to install Docker Desktop..."
    if (Try-InstallDockerWithWinget) {
        return $true
    }
    if (Try-InstallDockerWithChocolatey) {
        return $true
    }
    return $false
}

function Test-PostgresRuntimeAvailable {
    if (Test-DockerRuntimeAvailable) { return $true }
    return [bool](Find-PostgresBinDir)
}

function Ensure-PostgresRuntime {
    if (Test-DockerRuntimeAvailable) {
        return "docker"
    }

    if (Ensure-DockerRuntime) {
        return "docker"
    }

    Show-SubInfo "Docker unavailable. Trying PostgreSQL tools fallback..."

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        $wingetIds = @(
            "PostgreSQL.PostgreSQL",
            "PostgreSQL.PostgreSQL.16"
        )
        foreach ($id in $wingetIds) {
            try {
                winget install --id $id --exact --silent --accept-source-agreements --accept-package-agreements *> $null
                if (Test-PostgresRuntimeAvailable) {
                    return "winget ($id)"
                }
            } catch {
            }
        }
    }

    $choco = Get-Command choco -ErrorAction SilentlyContinue
    if ($choco) {
        try {
            choco install postgresql -y *> $null
            if (Test-PostgresRuntimeAvailable) {
                return "chocolatey"
            }
        } catch {
        }
    }

    return $null
}

# ── PostgresOnly Mode ────────────────────────────────────────────────

if ($PostgresOnly) {
    $result = Ensure-PostgresRuntime
    if (-not $result) {
        Write-Host "Failed to install Postgres runtime prerequisites." -ForegroundColor Red
        Write-Host "Install Docker Desktop (recommended) or PostgreSQL tools, then rerun setup." -ForegroundColor Yellow
        exit 1
    }
    exit 0
}

# ── Main Setup Flow ──────────────────────────────────────────────────

if (-not $NoBanner) {
    Show-Banner
}

$pythonMinMinor = 10
$pythonMaxMinor = 12

function Test-PythonVersionSupported {
    param(
        [string]$Command,
        [string[]]$PrefixArgs = @()
    )

    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        return $false
    }

    try {
        & $Command @PrefixArgs -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 and $pythonMinMinor <= sys.version_info.minor <= $pythonMaxMinor else 1)" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Get-PythonVersionString {
    param(
        [string]$Command,
        [string[]]$PrefixArgs = @()
    )

    try {
        return (& $Command @PrefixArgs -c "import platform; print(platform.python_version())")
    } catch {
        return $null
    }
}

function Get-SupportedPythonCandidate {
    $candidates = @(
        @{ command = "py"; prefix = @("-3.12") },
        @{ command = "py"; prefix = @("-3.11") },
        @{ command = "py"; prefix = @("-3.10") },
        @{ command = "python3.12"; prefix = @() },
        @{ command = "python3.11"; prefix = @() },
        @{ command = "python3.10"; prefix = @() },
        @{ command = "python"; prefix = @() }
    )

    foreach ($candidate in $candidates) {
        if (Test-PythonVersionSupported -Command $candidate.command -PrefixArgs $candidate.prefix) {
            return $candidate
        }
    }
    return $null
}

function Install-SupportedPython {
    Show-SubInfo "No Python 3.10-3.12 found. Attempting automatic install..."

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        foreach ($id in @("Python.Python.3.12", "Python.Python.3.11", "Python.Python.3.10")) {
            try {
                winget install --id $id --exact --silent --accept-source-agreements --accept-package-agreements *> $null
                if (Get-SupportedPythonCandidate) {
                    return $true
                }
            } catch {
            }
        }
    }

    $choco = Get-Command choco -ErrorAction SilentlyContinue
    if ($choco) {
        foreach ($pkg in @("python312", "python311", "python310", "python")) {
            try {
                choco install $pkg -y *> $null
                if (Get-SupportedPythonCandidate) {
                    return $true
                }
            } catch {
            }
        }
    }

    return $false
}

# ── Step 1: Detect Python ────────────────────────────────────────────

Show-Step "Detecting Python"

$pythonCandidate = Get-SupportedPythonCandidate
if (-not $pythonCandidate) {
    Install-SupportedPython | Out-Null
    $pythonCandidate = Get-SupportedPythonCandidate
}
if (-not $pythonCandidate) {
    Show-StepFail "Python 3.10-3.12 required"
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $detected = python --version 2>&1
        Show-SubInfo "Detected: $detected"
    }
    Show-SubInfo "Install Python 3.12 or 3.11 and rerun setup."
    exit 1
}

$pythonCommand = [string]$pythonCandidate.command
$pythonPrefixArgs = [string[]]$pythonCandidate.prefix
$pythonVersion = Get-PythonVersionString -Command $pythonCommand -PrefixArgs $pythonPrefixArgs
$pythonCommandLabel = if ($pythonPrefixArgs.Count -gt 0) {
    "$pythonCommand $($pythonPrefixArgs -join ' ')"
} else {
    $pythonCommand
}
Show-StepOK "$pythonVersion ($pythonCommandLabel)"

# ── Step 2: Detect Node.js ───────────────────────────────────────────

Show-Step "Detecting Node.js"

try {
    $nodeVersion = node --version 2>&1
    Show-StepOK "$nodeVersion"
} catch {
    Show-StepFail "not found"
    Show-SubInfo "Download from https://nodejs.org/"
    exit 1
}

# ── Step 3: Backend environment ──────────────────────────────────────

Show-Step "Preparing backend environment"

Push-Location backend

if (Test-Path "venv") {
    $venvPython = ".\venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Remove-Item -Recurse -Force "venv"
    } else {
        try {
            $venvVersionRaw = & $venvPython --version 2>&1
            $venvMatch = [regex]::Match($venvVersionRaw, '(\d+)\.(\d+)')
            $venvMajor = [int]$venvMatch.Groups[1].Value
            $venvMinor = [int]$venvMatch.Groups[2].Value
            if ($venvMajor -ne 3 -or $venvMinor -lt $pythonMinMinor -or $venvMinor -gt $pythonMaxMinor) {
                Remove-Item -Recurse -Force "venv"
            }
        } catch {
            if (Test-Path "venv") {
                Remove-Item -Recurse -Force "venv"
            }
        }
    }
}

if (-not (Test-Path "venv")) {
    & $pythonCommand @pythonPrefixArgs -m venv venv
}

& .\venv\Scripts\Activate.ps1

try {
    $activePythonVersion = python --version 2>&1
    $activeMatch = [regex]::Match($activePythonVersion, '(\d+)\.(\d+)')
    $activeMajor = [int]$activeMatch.Groups[1].Value
    $activeMinor = [int]$activeMatch.Groups[2].Value
    if ($activeMajor -ne 3 -or $activeMinor -lt $pythonMinMinor -or $activeMinor -gt $pythonMaxMinor) {
        Pop-Location
        Show-StepFail "venv Python $activePythonVersion unsupported"
        Show-SubInfo "Delete backend\venv and rerun setup."
        exit 1
    }
} catch {
    Pop-Location
    Show-StepFail "could not verify venv Python"
    exit 1
}

Show-StepOK "venv ready"

Pop-Location

# ── Step 4a: WeasyPrint native libs (Windows: GTK runtime) ───────────
# WeasyPrint powers the executive PDF report endpoint
# (services/reports/wallet_strategy_report.py).  On Windows it depends
# on the GTK 3 runtime (Pango / Cairo / GdkPixbuf).  We try winget /
# choco / msys2 in that order; any failure is logged and we keep going.
# The PDF endpoint returns a 503 with an actionable install hint when
# WeasyPrint can't initialize, so the rest of the API runs regardless.
Show-Step "Installing WeasyPrint native libs (GTK runtime)"
$weasyOk = $false
try {
    $gtkProbe = (& gtk-update-icon-cache --version 2>$null)
    if ($LASTEXITCODE -eq 0) { $weasyOk = $true }
} catch {}
if (-not $weasyOk) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        # Tobias Schönberg's GTK+ for Windows Runtime Environment is the
        # canonical Windows distribution of the GTK 3 runtime that ships
        # the libgobject/libpango/libcairo DLLs WeasyPrint depends on.
        # See https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer
        try {
            winget install --silent --accept-source-agreements --accept-package-agreements `
                --id tschoonj.GTKForWindows 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { $weasyOk = $true }
        } catch {}
    }
}
if (-not $weasyOk -and (Get-Command choco -ErrorAction SilentlyContinue)) {
    try {
        choco install -y --no-progress gtk-runtime 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $weasyOk = $true }
    } catch {}
}
if ($weasyOk) {
    Show-StepOK "GTK runtime detected"
} else {
    Show-StepOK "(GTK runtime not auto-installed - PDF report endpoint will surface install hint)"
}

# ── Step 4: Python dependencies ──────────────────────────────────────

Show-Step "Installing Python dependencies"

Push-Location backend
& .\venv\Scripts\Activate.ps1

python -m pip install --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Show-StepFail "pip upgrade failed"
    exit 1
}

python -m pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Show-StepFail "requirements.txt install failed"
    exit 1
}

python -c "import socksio" *> $null
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Show-StepFail "socksio missing"
    Show-SubInfo "Run: python -m pip install `"httpx[socks]>=0.27.0,<1.0`""
    exit 1
}

Show-StepOK

# ── Step 5: Trading dependencies ─────────────────────────────────────

Show-Step "Installing trading packages"

python -m pip install --quiet -r requirements-trading.txt
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Show-StepFail "requirements-trading.txt install failed"
    exit 1
}

python -c "import py_clob_client_v2, eth_account" *> $null
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Show-StepFail "trading imports missing"
    Show-SubInfo "Expected: py_clob_client_v2, eth_account"
    exit 1
}

Pop-Location
Show-StepOK

# ── Step 6: Frontend dependencies ────────────────────────────────────

Show-Step "Installing frontend packages"

Push-Location frontend

npm install --silent 2>$null
if ($LASTEXITCODE -ne 0) {
    npm install
}
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Show-StepFail "npm install failed"
    exit 1
}

Pop-Location
Show-StepOK

# ── Step 7: Launcher tooling ────────────────────────────────────────

Show-Step "Setting up launcher tooling"

$toolingInstallOk = $false
try {
    $originalCxxFlags = $env:CXXFLAGS
    $env:CXXFLAGS = "$($env:CXXFLAGS) /std:c++20".Trim()
    $ErrorActionPreference = "Continue"
    npm --prefix scripts/infra/tooling install --silent 2>$null
    if ($LASTEXITCODE -ne 0) {
        npm --prefix scripts/infra/tooling install 2>$null
    }
    $toolingInstallOk = ($LASTEXITCODE -eq 0)
} catch {
    $toolingInstallOk = $false
} finally {
    $env:CXXFLAGS = $originalCxxFlags
    $ErrorActionPreference = "Stop"
}

if ($toolingInstallOk) {
    try {
        $ErrorActionPreference = "Continue"
        node .\scripts\infra\tooling\check_powershell_syntax.mjs .\scripts\infra\run.ps1 .\scripts\infra\setup.ps1
        $ErrorActionPreference = "Stop"
        if ($LASTEXITCODE -ne 0) {
            Show-StepWarn "syntax check failed (non-fatal)"
        } else {
            Show-StepOK
        }
    } catch {
        $ErrorActionPreference = "Stop"
        Show-StepWarn "syntax check failed (non-fatal)"
    }
} else {
    Show-StepWarn "skipped (non-fatal)"
}

# ── Step 8: Postgres runtime ────────────────────────────────────────

# Create data directory
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
}

Show-Step "Verifying Postgres runtime"

$pgResult = Ensure-PostgresRuntime
if (-not $pgResult) {
    Show-StepFail "no runtime found"
    Show-SubInfo "Install Docker Desktop (recommended) or PostgreSQL tools, then rerun setup."
    exit 1
}

Show-StepOK $pgResult

# ── Write setup fingerprint ─────────────────────────────────────────

function Get-HashOrMissing {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return "missing" }
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$stamp = @{
    python_version = (& .\backend\venv\Scripts\python.exe -c "import platform; print(platform.python_version())")
    requirements_sha256 = Get-HashOrMissing "backend\requirements.txt"
    requirements_trading_sha256 = Get-HashOrMissing "backend\requirements-trading.txt"
    package_json_sha256 = Get-HashOrMissing "frontend\package.json"
    package_lock_sha256 = Get-HashOrMissing "frontend\package-lock.json"
    launcher_tools_package_json_sha256 = if ($toolingInstallOk) { Get-HashOrMissing "scripts\infra\tooling\package.json" } else { "skipped" }
    launcher_tools_package_lock_sha256 = if ($toolingInstallOk) { Get-HashOrMissing "scripts\infra\tooling\package-lock.json" } else { "skipped" }
}

$stamp | ConvertTo-Json | Set-Content -Path ".setup-stamp.json" -Encoding UTF8

# ── Completion ───────────────────────────────────────────────────────

Write-Host ""
Write-Host "    ---------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""
Write-Host "    Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "    Start the application:" -ForegroundColor White
Write-Host "      .\scripts\infra\run.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "    Or start services individually:" -ForegroundColor DarkGray
Write-Host '      Backend:  cd backend; .\venv\Scripts\Activate.ps1; uvicorn main:app --reload' -ForegroundColor DarkGray
Write-Host '      Frontend: cd frontend; npm run dev' -ForegroundColor DarkGray
Write-Host ""
Write-Host "    Endpoints:" -ForegroundColor White
Write-Host "      Frontend  http://localhost:3000" -ForegroundColor Cyan
Write-Host "      Backend   http://localhost:8000" -ForegroundColor Cyan
Write-Host "      API Docs  http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "    ---------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""
