# Homerun - Windows Run Script (TUI)
# Run: .\scripts\infra\run.ps1

$ErrorActionPreference = "Stop"

# Navigate to project root (grandparent of scripts\infra\)
Set-Location (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)))

$runServiceSmokeTest = $false
$tuiArgs = @()
foreach ($arg in $args) {
    if ($arg -eq "--services-smoke-test") {
        $runServiceSmokeTest = $true
    } else {
        $tuiArgs += $arg
    }
}

$postgresHost = if ($env:POSTGRES_HOST) { $env:POSTGRES_HOST } else { "127.0.0.1" }
$postgresPort = if ($env:POSTGRES_PORT) { [int]$env:POSTGRES_PORT } else { 5432 }
$postgresDb = if ($env:POSTGRES_DB) { $env:POSTGRES_DB } else { "homerun" }
$postgresUser = if ($env:POSTGRES_USER) { $env:POSTGRES_USER } else { "homerun" }
$postgresPassword = if ($env:POSTGRES_PASSWORD) { $env:POSTGRES_PASSWORD } else { "homerun" }
$postgresContainerName = if ($env:POSTGRES_CONTAINER_NAME) { $env:POSTGRES_CONTAINER_NAME } else { "homerun-postgres" }
$postgresImage = if ($env:POSTGRES_IMAGE) { $env:POSTGRES_IMAGE } else { "postgres:16-alpine" }
$postgresDataDir = if ($env:POSTGRES_DATA_DIR) { $env:POSTGRES_DATA_DIR } else { Join-Path (Get-Location).Path "data\postgres" }
$script:dockerComposeFilePath = Join-Path (Get-Location).Path "scripts\infra\docker-compose.infra.yml"
$script:dockerComposeProjectName = if ($env:HOMERUN_COMPOSE_PROJECT) { $env:HOMERUN_COMPOSE_PROJECT } else { "homerun" }
$script:postgresPort = [int]$postgresPort

$script:postgresStartedByScript = $false
$script:postgresStartMode = ""
$script:postgresDockerCreatedByScript = $false
$script:lastPostgresContainerEngine = "docker"
$script:lastPostgresDockerPublishedPort = $null
$script:lastPostgresDockerError = $null
$script:databaseUrlWasProvided = [bool]$env:DATABASE_URL

function Ensure-ConsoleViewport {
    param(
        [int]$MinCols = 140,
        [int]$MinRows = 45,
        [int]$MinBufferRows = 60
)

    return
}

Ensure-ConsoleViewport

function Test-TcpPort {
    param(
        [string]$TargetHost,
        [int]$Port
    )

    $client = $null
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $client.ReceiveTimeout = 500
        $client.SendTimeout = 500
        $client.Connect($TargetHost, $Port)
        return $true
    } catch {
        return $false
    } finally {
        if ($client) { $client.Dispose() }
    }
}

function Test-NpcapLoopbackInterference {
    <#
    .SYNOPSIS
    Detect and disable the Npcap Loopback Adapter if present.

    The Npcap Loopback Adapter (installed by Wireshark/Nmap) can interfere with
    TCP connections on 127.0.0.1, causing postgres and other services to appear
    to listen but drop all incoming SYN-ACK packets.  This function checks if
    the adapter is present and enabled, and attempts to disable it (requires
    elevation).  If elevation fails, it warns the user.
    #>
    $adapter = Get-NetAdapter -Name "Npcap Loopback Adapter" -ErrorAction SilentlyContinue
    if (-not $adapter -or $adapter.Status -ne "Up") {
        return  # Not present or already disabled
    }

    Write-Host "Npcap Loopback Adapter detected (enabled). This can block local database connections." -ForegroundColor Yellow
    Write-Host "Attempting to disable it..." -ForegroundColor Yellow

    try {
        # Try directly (works if running elevated)
        Disable-NetAdapter -Name "Npcap Loopback Adapter" -Confirm:$false -ErrorAction Stop
        Write-Host "Npcap Loopback Adapter disabled." -ForegroundColor Green
        return
    } catch {}

    # Try self-elevation
    try {
        $tmpScript = Join-Path $env:TEMP "homerun_disable_npcap.ps1"
        "Disable-NetAdapter -Name 'Npcap Loopback Adapter' -Confirm:`$false -ErrorAction Stop" | Set-Content $tmpScript -Encoding UTF8
        $proc = Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$tmpScript`"" -Wait -PassThru
        Remove-Item $tmpScript -Force -ErrorAction SilentlyContinue
        if ($proc.ExitCode -eq 0) {
            Write-Host "Npcap Loopback Adapter disabled." -ForegroundColor Green
            return
        }
    } catch {}

    Write-Host "Could not disable the Npcap Loopback Adapter automatically." -ForegroundColor Red
    Write-Host "Please disable it manually: Network Settings > Change adapter options > right-click 'Npcap Loopback Adapter' > Disable" -ForegroundColor Yellow
    Write-Host "Or run in an elevated terminal: Disable-NetAdapter -Name 'Npcap Loopback Adapter' -Confirm:`$false" -ForegroundColor Yellow
}

function Wait-ForService {
    param(
        [string]$TargetHost,
        [int]$Port,
        [int]$TimeoutSeconds = 10
    )

    $attempts = [Math]::Max(1, [int]([Math]::Ceiling(($TimeoutSeconds * 1000) / 250)))
    for ($i = 0; $i -lt $attempts; $i++) {
        if (Test-TcpPort -TargetHost $TargetHost -Port $Port) {
            return $true
        }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

$inWindowsTerminal = [bool]$env:WT_SESSION
$allowWindowsTerminal = @("1", "true", "yes", "on") -contains (($env:HOMERUN_TUI_ALLOW_WT | Out-String).Trim().ToLowerInvariant())
$alreadyRelaunched = @("1", "true", "yes", "on") -contains (($env:HOMERUN_CONHOST_RELAUNCHED | Out-String).Trim().ToLowerInvariant())

if ($inWindowsTerminal -and -not $allowWindowsTerminal -and -not $alreadyRelaunched) {
    [Environment]::SetEnvironmentVariable("HOMERUN_CONHOST_RELAUNCHED", "1", "Process")
    $relaunchArgs = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $MyInvocation.MyCommand.Path
    ) + $args
    Start-Process -FilePath "powershell.exe" -WorkingDirectory (Get-Location).Path -ArgumentList $relaunchArgs | Out-Null
    exit 0
}

if ($inWindowsTerminal -and -not $allowWindowsTerminal) {
    [Environment]::SetEnvironmentVariable("HOMERUN_TUI_SAFE_CONSOLE", "1", "Process")
}

function Get-ListeningProcessId {
    param(
        [string]$TargetHost,
        [int]$Port
    )

    try {
        $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    } catch {
        return $null
    }

    if (-not $listeners) {
        return $null
    }

    foreach ($listener in $listeners) {
        $localAddress = ($listener.LocalAddress | Out-String).Trim()
        if ($TargetHost -and $TargetHost -ne "0.0.0.0") {
            if ($localAddress -ne $TargetHost -and $localAddress -ne "0.0.0.0" -and $localAddress -ne "::" -and $localAddress -ne "::0" -and $localAddress -ne "::1") {
                continue
            }
        }
        if ($listener.OwningProcess) {
            return [int]$listener.OwningProcess
        }
    }

    return $null
}

function Wait-ForPortToClose {
    param(
        [string]$TargetHost,
        [int]$Port,
        [int]$TimeoutSeconds = 10
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $listenerPid = Get-ListeningProcessId -TargetHost $TargetHost -Port $Port
        if (-not $listenerPid) {
            return $true
        }
        Start-Sleep -Milliseconds 250
    }

    return $false
}

function Test-PortBindConflictError {
    param(
        [string]$Output,
        [int]$Port
    )

    if (-not $Output) {
        return $false
    }

    return (($Output -match "(?i)ports are not available") -and ($Output -match "(?i):$Port\b"))
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

function Get-PostgresBinaryMajorVersion {
    param([string]$PostgresBinDir)

    $postgresExe = Join-Path $PostgresBinDir "postgres.exe"
    if (-not (Test-Path $postgresExe)) {
        return $null
    }

    try {
        $versionOutput = (& $postgresExe --version 2>&1 | Out-String).Trim()
    } catch {
        return $null
    }

    $versionMatch = [regex]::Match($versionOutput, '(\d+)\.(\d+)')
    if (-not $versionMatch.Success) {
        return $null
    }

    try {
        return [int]$versionMatch.Groups[1].Value
    } catch {
        return $null
    }
}

function Move-PostgresDataDirToStale {
    param([string]$DataDir)

    if (-not (Test-Path $DataDir)) {
        return
    }

    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $staleDir = "${DataDir}.stale.${timestamp}"
    try {
        Move-Item -Path $DataDir -Destination $staleDir -Force -ErrorAction Stop
        Write-Host "Moved incompatible Postgres data directory to $staleDir" -ForegroundColor Yellow
    } catch {
        # Fall back to best-effort cleanup if rename fails.
        try {
            $entries = Get-ChildItem -Path $DataDir -Force -ErrorAction SilentlyContinue
            foreach ($entry in $entries) {
                Remove-Item -Path $entry.FullName -Recurse -Force -ErrorAction SilentlyContinue
            }
            Write-Host "Cleared stale Postgres data directory contents at $DataDir" -ForegroundColor Yellow
        } catch {
        }
    }
}

function Show-PostgresInitLogs {
    param(
        [string]$InitStdoutPath,
        [string]$InitStderrPath
    )

    if (Test-Path $InitStdoutPath) {
        Write-Host "Postgres init stdout (tail):" -ForegroundColor Yellow
        Get-Content -Path $InitStdoutPath -Tail 20 -ErrorAction SilentlyContinue | ForEach-Object {
            Write-Host "  $_" -ForegroundColor DarkYellow
        }
    }
    if (Test-Path $InitStderrPath) {
        Write-Host "Postgres init stderr (tail):" -ForegroundColor Yellow
        Get-Content -Path $InitStderrPath -Tail 20 -ErrorAction SilentlyContinue | ForEach-Object {
            Write-Host "  $_" -ForegroundColor DarkYellow
        }
    }
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

function Invoke-DockerCommand {
    param([string[]]$Arguments)

    $result = [ordered]@{
        ExitCode = 1
        Output = ""
    }

    if (-not (Ensure-DockerCommand)) {
        $result.Output = "Docker CLI unavailable in current shell."
        return [pscustomobject]$result
    }

    $nativePrefExists = $false
    $previousNativePref = $null
    try {
        if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
            $nativePrefExists = $true
            $previousNativePref = $PSNativeCommandUseErrorActionPreference
            $PSNativeCommandUseErrorActionPreference = $false
        }
    } catch {
    }

    try {
        $output = (& docker @Arguments 2>&1 | Out-String).Trim()
        $result.ExitCode = $LASTEXITCODE
        $result.Output = $output
    } catch {
        $result.ExitCode = if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { $LASTEXITCODE } else { 1 }
        $result.Output = $_.Exception.Message
    } finally {
        if ($nativePrefExists) {
            $PSNativeCommandUseErrorActionPreference = $previousNativePref
        }
    }

    return [pscustomobject]$result
}

function Test-DockerComposeAvailable {
    if (-not (Ensure-DockerCommand)) {
        return $false
    }

    $composeVersion = Invoke-DockerCommand -Arguments @("compose", "version")
    return ($composeVersion.ExitCode -eq 0)
}

function Invoke-DockerComposeCommand {
    param(
        [string[]]$Arguments,
        [hashtable]$EnvironmentOverrides = @{}
    )

    $result = [ordered]@{
        ExitCode = 1
        Output = ""
    }

    if (-not (Test-Path $script:dockerComposeFilePath)) {
        $result.Output = "Compose file missing: $script:dockerComposeFilePath"
        return [pscustomobject]$result
    }
    if (-not (Test-DockerComposeAvailable)) {
        $result.Output = "docker compose is unavailable."
        return [pscustomobject]$result
    }

    $previousValues = @{}
    foreach ($entry in $EnvironmentOverrides.GetEnumerator()) {
        $key = [string]$entry.Key
        $previousValues[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, [string]$entry.Value, "Process")
    }

    try {
        $composeArgs = @("compose", "-p", $script:dockerComposeProjectName, "-f", $script:dockerComposeFilePath) + $Arguments
        $invoked = Invoke-DockerCommand -Arguments $composeArgs
        $result.ExitCode = $invoked.ExitCode
        $result.Output = $invoked.Output
    } finally {
        foreach ($entry in $previousValues.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable([string]$entry.Key, $entry.Value, "Process")
        }
    }

    return [pscustomobject]$result
}

function Test-ContainerNameConflict {
    param(
        [string]$Output,
        [string]$ContainerName
    )

    if (-not $Output) {
        return $false
    }
    $escapedName = [regex]::Escape($ContainerName)
    return (($Output -match "(?i)already in use by container") -and ($Output -match "(?i)/?$escapedName"))
}

function Remove-ContainerIfExists {
    param([string]$ContainerName)

    $inspect = Invoke-DockerCommand -Arguments @("container", "inspect", $ContainerName)
    if ($inspect.ExitCode -ne 0) {
        return $true
    }

    $stop = Invoke-DockerCommand -Arguments @("stop", $ContainerName)
    if ($stop.ExitCode -ne 0) {
        $kill = Invoke-DockerCommand -Arguments @("kill", $ContainerName)
        if ($kill.ExitCode -ne 0) {
            return $false
        }
    }

    $rm = Invoke-DockerCommand -Arguments @("rm", $ContainerName)
    return ($rm.ExitCode -eq 0)
}

function Get-DockerPublishedHostPort {
    param(
        [string]$ContainerName,
        [string]$ContainerPort
    )

    if (-not (Ensure-DockerCommand)) {
        return $null
    }

    $portResult = Invoke-DockerCommand -Arguments @("port", $ContainerName, $ContainerPort)
    if ($portResult.ExitCode -ne 0) {
        return $null
    }

    $output = ($portResult.Output | Out-String).Trim()
    if (-not $output) {
        return $null
    }

    foreach ($line in ($output -split "`r?`n")) {
        $match = [regex]::Match(($line | Out-String).Trim(), ':(\d+)\s*$')
        if ($match.Success) {
            return [int]$match.Groups[1].Value
        }
    }

    return $null
}

function Wait-ForDockerPublishedHostPort {
    param(
        [string]$ContainerName,
        [string]$ContainerPort,
        [int]$TimeoutSeconds = 20
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $publishedPort = Get-DockerPublishedHostPort -ContainerName $ContainerName -ContainerPort $ContainerPort
        if ($publishedPort) {
            return [int]$publishedPort
        }
        Start-Sleep -Milliseconds 500
    }

    return $null
}

function Test-DockerRuntimeAvailable {
    $dockerInfo = Invoke-DockerCommand -Arguments @("info")
    return ($dockerInfo.ExitCode -eq 0)
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

function Test-PostgresRuntimeAvailable {
    if (Test-DockerRuntimeAvailable) { return $true }
    return [bool](Find-PostgresBinDir)
}

function Ensure-PostgresRuntime {
    if (Test-PostgresRuntimeAvailable) {
        return $true
    }

    Write-Host "Postgres runtime missing; invoking setup postgres bootstrap..." -ForegroundColor Cyan
    try {
        & .\scripts\infra\setup.ps1 -PostgresOnly
        return (Test-PostgresRuntimeAvailable)
    } catch {
        return $false
    }
}

function Show-PostgresDockerDiagnostics {
    param([string]$ContainerName)

    if (-not (Ensure-DockerCommand)) {
        return
    }

    $statusResult = Invoke-DockerCommand -Arguments @("inspect", "-f", "{{.State.Status}}", $ContainerName)
    if ($statusResult.ExitCode -eq 0 -and $statusResult.Output) {
        Write-Host "Postgres container status: $($statusResult.Output)" -ForegroundColor Yellow
    }

    $publishedPort = Get-DockerPublishedHostPort -ContainerName $ContainerName -ContainerPort "5432/tcp"
    if ($publishedPort) {
        Write-Host "Postgres container published port: $publishedPort" -ForegroundColor Yellow
    }

    $logsResult = Invoke-DockerCommand -Arguments @("logs", "--tail", "40", $ContainerName)
    if ($logsResult.Output) {
        Write-Host "Postgres container logs (tail):" -ForegroundColor Yellow
        $logsResult.Output -split "`r?`n" | ForEach-Object {
            Write-Host "  $_" -ForegroundColor DarkYellow
        }
    }
}

function Start-PostgresDocker {
    param(
        [string]$PgHost,
        [int]$Port,
        [string]$Db,
        [string]$User,
        [string]$Password,
        [string]$ContainerName,
        [string]$Image,
        [string]$DataDir
    )

    $script:lastPostgresDockerError = $null
    $script:lastPostgresDockerPublishedPort = $null
    $script:lastPostgresContainerEngine = "docker"

    if (-not (Ensure-DockerCommand)) {
        $script:lastPostgresDockerError = "Docker CLI unavailable in current shell."
        return $false
    }

    $dockerInfo = Invoke-DockerCommand -Arguments @("info")
    if ($dockerInfo.ExitCode -ne 0) {
        $script:lastPostgresDockerError = if ($dockerInfo.Output) { $dockerInfo.Output } else { "docker info returned exit code $($dockerInfo.ExitCode)." }
        return $false
    }

    $composeAvailable = (Test-Path $script:dockerComposeFilePath) -and (Test-DockerComposeAvailable)
    if ($composeAvailable) {
        $composeResult = Invoke-DockerComposeCommand `
            -Arguments @("up", "-d", "postgres") `
            -EnvironmentOverrides @{
                "POSTGRES_HOST" = $PgHost
                "POSTGRES_PORT" = "$Port"
                "POSTGRES_DB" = $Db
                "POSTGRES_USER" = $User
                "POSTGRES_PASSWORD" = $Password
                "POSTGRES_IMAGE" = $Image
                "POSTGRES_CONTAINER_NAME" = $ContainerName
            }
        if ($composeResult.ExitCode -eq 0) {
            $script:lastPostgresContainerEngine = "docker-compose"
            $script:postgresDockerCreatedByScript = $true
            $publishedPort = Wait-ForDockerPublishedHostPort -ContainerName $ContainerName -ContainerPort "5432/tcp" -TimeoutSeconds 30
            if ($publishedPort) {
                $script:lastPostgresDockerPublishedPort = [int]$publishedPort
            } else {
                $script:lastPostgresDockerPublishedPort = [int]$Port
            }
            return $true
        }

        if (Test-PostgresDockerListenerOwned -ContainerName $ContainerName -Port $Port) {
            $script:lastPostgresContainerEngine = "docker-compose"
            $script:postgresDockerCreatedByScript = $true
            $script:lastPostgresDockerPublishedPort = [int]$Port
            return $true
        }

        if (Test-ContainerNameConflict -Output $composeResult.Output -ContainerName $ContainerName) {
            if (Remove-ContainerIfExists -ContainerName $ContainerName) {
                $composeRetry = Invoke-DockerComposeCommand `
                    -Arguments @("up", "-d", "postgres") `
                    -EnvironmentOverrides @{
                        "POSTGRES_HOST" = $PgHost
                        "POSTGRES_PORT" = "$Port"
                        "POSTGRES_DB" = $Db
                        "POSTGRES_USER" = $User
                        "POSTGRES_PASSWORD" = $Password
                        "POSTGRES_IMAGE" = $Image
                        "POSTGRES_CONTAINER_NAME" = $ContainerName
                    }
                if ($composeRetry.ExitCode -eq 0) {
                    $script:lastPostgresContainerEngine = "docker-compose"
                    $script:postgresDockerCreatedByScript = $true
                    $publishedPort = Wait-ForDockerPublishedHostPort -ContainerName $ContainerName -ContainerPort "5432/tcp" -TimeoutSeconds 30
                    if ($publishedPort) {
                        $script:lastPostgresDockerPublishedPort = [int]$publishedPort
                    } else {
                        $script:lastPostgresDockerPublishedPort = [int]$Port
                    }
                    return $true
                }
                if (Test-PostgresDockerListenerOwned -ContainerName $ContainerName -Port $Port) {
                    $script:lastPostgresContainerEngine = "docker-compose"
                    $script:postgresDockerCreatedByScript = $true
                    $script:lastPostgresDockerPublishedPort = [int]$Port
                    return $true
                }
                if ($composeRetry.Output) {
                    $script:lastPostgresDockerError = $composeRetry.Output
                }
            }
        } elseif ($composeResult.Output) {
            $script:lastPostgresDockerError = $composeResult.Output
        }

        $runningResult = Invoke-DockerCommand -Arguments @("inspect", "-f", "{{.State.Running}}", $ContainerName)
        if ($runningResult.ExitCode -eq 0) {
            $isRunning = ((($runningResult.Output | Out-String).Trim().ToLowerInvariant()) -eq "true")
            if ($isRunning) {
                $publishedPort = Wait-ForDockerPublishedHostPort -ContainerName $ContainerName -ContainerPort "5432/tcp" -TimeoutSeconds 20
                if ($publishedPort) {
                    $script:lastPostgresContainerEngine = "docker-compose"
                    $script:postgresDockerCreatedByScript = $true
                    $script:lastPostgresDockerPublishedPort = [int]$publishedPort
                    return $true
                }
            }
        }

        return $false
    }

    $imageInspect = Invoke-DockerCommand -Arguments @("image", "inspect", $Image)
    if ($imageInspect.ExitCode -ne 0) {
        Write-Host "Docker image '$Image' not present locally. Pulling..." -ForegroundColor Yellow
        $pullResult = Invoke-DockerCommand -Arguments @("pull", $Image)
        if ($pullResult.ExitCode -ne 0) {
            if ($pullResult.Output) {
                $script:lastPostgresDockerError = $pullResult.Output
            } else {
                $script:lastPostgresDockerError = "Failed to pull Docker image '$Image' (exit code $($pullResult.ExitCode))."
            }
            return $false
        }
    }

    try {
        New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    } catch {
        $script:lastPostgresDockerError = "Failed to create Postgres data directory '$DataDir': $($_.Exception.Message)"
        return $false
    }

    $runResult = Invoke-DockerCommand -Arguments @(
        "run",
        "--name", $ContainerName,
        "--detach",
        "--publish", "${PgHost}:${Port}:5432",
        "--env", "POSTGRES_DB=$Db",
        "--env", "POSTGRES_USER=$User",
        "--env", "POSTGRES_PASSWORD=$Password",
        "--volume", "${DataDir}:/var/lib/postgresql/data",
        $Image
    )
    if ($runResult.ExitCode -eq 0) {
        $script:postgresDockerCreatedByScript = $true
        $script:lastPostgresDockerPublishedPort = [int]$Port
        return $true
    }
    if ($runResult.Output) {
        $script:lastPostgresDockerError = $runResult.Output
    } else {
        $script:lastPostgresDockerError = "docker run failed with exit code $($runResult.ExitCode)."
    }
    return $false
}

function Start-PostgresLocal {
    param(
        [string]$PgHost,
        [int]$Port,
        [string]$User,
        [string]$DataDir
    )

    $binDir = Find-PostgresBinDir
    if (-not $binDir) {
        return $false
    }

    $initdbPath = Join-Path $binDir "initdb.exe"
    $pgctlPath = Join-Path $binDir "pg_ctl.exe"
    if (-not (Test-Path $initdbPath) -or -not (Test-Path $pgctlPath)) {
        return $false
    }

    try {
        New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    } catch {
        return $false
    }

    $binaryMajorVersion = Get-PostgresBinaryMajorVersion -PostgresBinDir $binDir
    $pgVersionPath = Join-Path $DataDir "PG_VERSION"
    $pgInitLogPath = Join-Path $DataDir "initdb.log"
    $pgInitErrLogPath = Join-Path $DataDir "initdb.err.log"
    $pgServerLogPath = Join-Path $DataDir "postgresql.log"

    if (Test-Path $pgVersionPath) {
        try {
            $existingMajorText = (Get-Content -Path $pgVersionPath -TotalCount 1 -ErrorAction Stop | Out-String).Trim()
            $existingMajorVersion = [int]$existingMajorText
            if (($null -ne $binaryMajorVersion) -and ($existingMajorVersion -ne $binaryMajorVersion)) {
                Write-Host "Postgres data dir version $existingMajorVersion is incompatible with installed Postgres $binaryMajorVersion. Reinitializing local data dir..." -ForegroundColor Yellow
                Move-PostgresDataDirToStale -DataDir $DataDir
                New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
                $pgVersionPath = Join-Path $DataDir "PG_VERSION"
            }
        } catch {
        }
    }

    if (-not (Test-Path $pgVersionPath)) {
        try {
            $existingEntries = @(Get-ChildItem -Path $DataDir -Force -ErrorAction SilentlyContinue)
            if ($existingEntries.Count -gt 0) {
                Move-PostgresDataDirToStale -DataDir $DataDir
                New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
            }

            $initArgs = @(
                "-D", $DataDir,
                "-U", $User,
                "--encoding=UTF8",
                "-A", "trust",
                "--no-instructions"
            )
            $initProc = Start-Process -FilePath $initdbPath -ArgumentList $initArgs -RedirectStandardOutput $pgInitLogPath -RedirectStandardError $pgInitErrLogPath -WindowStyle Hidden -PassThru
            $null = Wait-Process -Id $initProc.Id -Timeout 120 -ErrorAction SilentlyContinue
            if (-not $initProc.HasExited) {
                Stop-Process -Id $initProc.Id -Force -ErrorAction SilentlyContinue
                Write-Host "Postgres initialization timed out after 120s. See: $pgInitLogPath, $pgInitErrLogPath" -ForegroundColor Yellow
                Show-PostgresInitLogs -InitStdoutPath $pgInitLogPath -InitStderrPath $pgInitErrLogPath
                return $false
            }
            $initProc.Refresh()
            if ($initProc.ExitCode -ne 0) {
                Write-Host "Postgres initialization failed. See: $pgInitLogPath, $pgInitErrLogPath" -ForegroundColor Yellow
                Show-PostgresInitLogs -InitStdoutPath $pgInitLogPath -InitStderrPath $pgInitErrLogPath
                return $false
            }

            @"
local all all trust
host all all 127.0.0.1/32 trust
host all all ::1/128 trust
"@ | Set-Content -Path (Join-Path $DataDir "pg_hba.conf") -Encoding UTF8

            Add-Content -Path (Join-Path $DataDir "postgresql.conf") -Value "listen_addresses = '$PgHost'"
            Add-Content -Path (Join-Path $DataDir "postgresql.conf") -Value "port = $Port"
            Add-Content -Path (Join-Path $DataDir "postgresql.conf") -Value "max_connections = 200"
        } catch {
            return $false
        }
    }

    # On Windows a hard-killed Postgres can leave stale shared-memory
    # that causes the first start to fail immediately; we retry once.
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        try {
            & $pgctlPath -D $DataDir -l $pgServerLogPath -o "-h $PgHost -p $Port" -w -t 20 start *> $null
            if ($LASTEXITCODE -eq 0) { return $true }
        } catch {}

        if ($attempt -eq 1) {
            # Stale shared-memory from a crashed process is the most common
            # reason for failure.  Remove the PID file and pause to let the
            # OS reclaim the segment before the second attempt.
            $stalePid = Join-Path $DataDir "postmaster.pid"
            Remove-Item -Path $stalePid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 3
        }
    }

    if (Test-Path $pgServerLogPath) {
        Write-Host "Postgres startup failed. Recent log output:" -ForegroundColor Yellow
        Get-Content -Path $pgServerLogPath -Tail 20 -ErrorAction SilentlyContinue | ForEach-Object {
            Write-Host "  $_" -ForegroundColor DarkYellow
        }
    }
    return $false
}

function Ensure-PostgresFirewallRule {
    <#
    .SYNOPSIS
    Ensure an inbound firewall allow rule exists for postgres.exe.

    On Windows 11, when postgres.exe first listens on a port, the firewall may
    silently block inbound connections if the UAC allow dialog was not shown or
    was dismissed (common when started from a background script).  This function
    adds an explicit allow rule for the binary.  Requires elevation; fails
    silently without it.
    #>
    param([string]$PostgresBinDir)

    $pgExe = Join-Path $PostgresBinDir "postgres.exe"
    if (-not (Test-Path $pgExe)) { return }

    $ruleName = "Homerun - PostgreSQL Server"

    # Check if rule already exists
    $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if ($existing) { return }

    try {
        New-NetFirewallRule `
            -DisplayName $ruleName `
            -Direction Inbound `
            -Action Allow `
            -Protocol TCP `
            -Program $pgExe `
            -Profile Any `
            -ErrorAction Stop | Out-Null
    } catch {
        # Not elevated - fall back to netsh which sometimes works or just warn
        try {
            netsh advfirewall firewall add rule name="$ruleName" dir=in action=allow protocol=TCP program="$pgExe" profile=any *> $null
        } catch {
            Write-Host "Warning: Could not add firewall rule for PostgreSQL. If connections fail, run as Administrator once or manually allow '$pgExe' in Windows Firewall." -ForegroundColor Yellow
        }
    }
}

function Test-PostgresDockerListenerOwned {
    param(
        [string]$ContainerName,
        [int]$Port
    )

    if (-not (Ensure-DockerCommand)) {
        return $false
    }

    $inspectResult = Invoke-DockerCommand -Arguments @("container", "inspect", $ContainerName)
    if ($inspectResult.ExitCode -ne 0) {
        return $false
    }

    $runningResult = Invoke-DockerCommand -Arguments @("inspect", "-f", "{{.State.Running}}", $ContainerName)
    if ($runningResult.ExitCode -ne 0) {
        return $false
    }
    if ((($runningResult.Output | Out-String).Trim().ToLowerInvariant()) -ne "true") {
        return $false
    }

    $publishedPort = Get-DockerPublishedHostPort -ContainerName $ContainerName -ContainerPort "5432/tcp"
    if (-not $publishedPort) {
        return $false
    }
    return ([int]$publishedPort -eq [int]$Port)
}

function Get-RunningDockerPostgresPort {
    param([string]$ContainerName)

    if (-not (Ensure-DockerCommand)) {
        return $null
    }

    $runningResult = Invoke-DockerCommand -Arguments @("inspect", "-f", "{{.State.Running}}", $ContainerName)
    if ($runningResult.ExitCode -ne 0) {
        return $null
    }
    if (((($runningResult.Output | Out-String).Trim().ToLowerInvariant()) -ne "true")) {
        return $null
    }

    $publishedPort = Get-DockerPublishedHostPort -ContainerName $ContainerName -ContainerPort "5432/tcp"
    if (-not $publishedPort) {
        return $null
    }
    return [int]$publishedPort
}

function Test-LocalPostgresListenerOwned {
    param(
        [string]$DataDir,
        [int]$Port
    )

    $pidPath = Join-Path $DataDir "postmaster.pid"
    if (-not (Test-Path $pidPath)) {
        return $false
    }

    try {
        $lines = Get-Content -Path $pidPath -ErrorAction Stop
        if ($lines.Count -lt 4) { return $false }
        $pidValue = ($lines[0] | Out-String).Trim()
        $portValue = ($lines[3] | Out-String).Trim()
        if (-not $pidValue -or -not $portValue) { return $false }
        $proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
        if (-not $proc) { return $false }
        return ($portValue -eq "$Port")
    } catch {
        return $false
    }
}

function Get-ParentProcess {
    param([int]$ProcessId)

    try {
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
        if (-not $processInfo) {
            return $null
        }
        $parentId = [int]$processInfo.ParentProcessId
        if ($parentId -le 0) {
            return $null
        }
        return Get-Process -Id $parentId -ErrorAction SilentlyContinue
    } catch {
        return $null
    }
}

function Stop-ConflictingPostgresListener {
    param(
        [string]$PgHost,
        [int]$Port,
        [string]$ContainerName,
        [string]$DataDir
    )

    if (Test-PostgresDockerListenerOwned -ContainerName $ContainerName -Port $Port) {
        return $true
    }

    try {
        $postgresServices = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue
        foreach ($svc in $postgresServices) {
            if ($svc -and $svc.Status -eq "Running") {
                Stop-Service -Name $svc.Name -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
    }

    if (Test-LocalPostgresListenerOwned -DataDir $DataDir -Port $Port) {
        $binDir = Find-PostgresBinDir
        if ($binDir) {
            $pgctlPath = Join-Path $binDir "pg_ctl.exe"
            if (Test-Path $pgctlPath) {
                try {
                    & $pgctlPath -D $DataDir -m fast -w stop *> $null
                } catch {
                }
            }
        }
    }

    if (Wait-ForPortToClose -TargetHost $PgHost -Port $Port -TimeoutSeconds 6) {
        return $true
    }

    $listenerPid = Get-ListeningProcessId -TargetHost $PgHost -Port $Port
    if (-not $listenerPid) {
        return $true
    }

    $proc = Get-Process -Id $listenerPid -ErrorAction SilentlyContinue
    if (-not $proc) {
        return (Wait-ForPortToClose -TargetHost $PgHost -Port $Port -TimeoutSeconds 2)
    }

    $processName = ($proc.ProcessName | Out-String).Trim().ToLowerInvariant()
    if ($processName -notmatch "postgres") {
        return $false
    }

    $parentProcess = Get-ParentProcess -ProcessId $listenerPid
    $stopProcessIds = [System.Collections.Generic.List[int]]::new()
    if ($parentProcess) {
        $parentName = ($parentProcess.ProcessName | Out-String).Trim().ToLowerInvariant()
        if ($parentName -match "pg_ctl") {
            $stopProcessIds.Add([int]$parentProcess.Id)
        }
    }
    $stopProcessIds.Add([int]$listenerPid)

    try {
        foreach ($processId in ($stopProcessIds | Select-Object -Unique)) {
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    } catch {
        return $false
    }

    return (Wait-ForPortToClose -TargetHost $PgHost -Port $Port -TimeoutSeconds 4)
}

function Test-LauncherPostgresListenerOwned {
    param(
        [string]$ContainerName,
        [string]$DataDir,
        [int]$Port
    )

    if (Test-PostgresDockerListenerOwned -ContainerName $ContainerName -Port $Port) {
        return $true
    }
    return (Test-LocalPostgresListenerOwned -DataDir $DataDir -Port $Port)
}

function Get-AvailablePostgresPort {
    param(
        [string]$PgHost,
        [int]$StartPort
    )

    for ($port = $StartPort; $port -le ($StartPort + 32); $port++) {
        if (-not (Test-TcpPort -TargetHost $PgHost -Port $port)) {
            return $port
        }
    }
    return $null
}

function Get-RunningLocalPostgresPort {
    param(
        [string]$DataDir
    )

    $pidPath = Join-Path $DataDir "postmaster.pid"
    if (-not (Test-Path $pidPath)) {
        return $null
    }

    try {
        $lines = Get-Content -Path $pidPath -ErrorAction Stop
        if ($lines.Count -lt 4) { return $null }
        $pidValue = ($lines[0] | Out-String).Trim()
        $portValue = ($lines[3] | Out-String).Trim()
        if (-not $pidValue -or -not $portValue) { return $null }
        $proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
        if (-not $proc) {
            # Stale postmaster.pid from crashed process - clean it up
            Remove-Item -Path $pidPath -Force -ErrorAction SilentlyContinue
            return $null
        }
        return [int]$portValue
    } catch {
        return $null
    }
}

function Ensure-Postgres {
    param(
        [string]$PgHost,
        [int]$Port,
        [string]$Db,
        [string]$User,
        [string]$Password,
        [string]$ContainerName,
        [string]$Image,
        [string]$DataDir
    )
    $script:postgresPort = [int]$Port

    # Check if our local Postgres is already running (on any port)
    $runningPort = Get-RunningLocalPostgresPort -DataDir $DataDir
    if ($runningPort) {
        $script:postgresPort = [int]$runningPort
        # Ensure firewall rule exists even for already-running instances
        $pgBinDir = Find-PostgresBinDir
        if ($pgBinDir) { Ensure-PostgresFirewallRule -PostgresBinDir $pgBinDir }
        Write-Host "Postgres already running on ${PgHost}:${runningPort}" -ForegroundColor Green
        return
    }

    $runningDockerPort = Get-RunningDockerPostgresPort -ContainerName $ContainerName
    if ($runningDockerPort) {
        $script:postgresPort = [int]$runningDockerPort
        Write-Host "Postgres already running on ${PgHost}:${runningDockerPort}" -ForegroundColor Green
        return
    }

    # Check if our Docker container is already running on the requested port
    if (Test-TcpPort -TargetHost $PgHost -Port $Port) {
        if (Test-PostgresDockerListenerOwned -ContainerName $ContainerName -Port $Port) {
            $script:postgresPort = [int]$Port
            Write-Host "Postgres already running on ${PgHost}:${Port}" -ForegroundColor Green
            return
        }

        Write-Host "Postgres port ${Port} is occupied by a non-launcher listener. Attempting to reclaim it..." -ForegroundColor Yellow
        if (Stop-ConflictingPostgresListener -PgHost $PgHost -Port $Port -ContainerName $ContainerName -DataDir $DataDir) {
            if (-not (Test-TcpPort -TargetHost $PgHost -Port $Port)) {
                Write-Host "Reclaimed Postgres port ${Port}." -ForegroundColor Green
            }
        }

        if (Test-TcpPort -TargetHost $PgHost -Port $Port) {
            $alternatePort = Get-AvailablePostgresPort -PgHost $PgHost -StartPort ($Port + 1)
            if (-not $alternatePort) {
                Write-Host "Port ${Port} is occupied and no alternate Postgres port is available." -ForegroundColor Red
                Write-Host "Set DATABASE_URL manually or free a local port, then rerun." -ForegroundColor Yellow
                exit 1
            }

            Write-Host "Port ${Port} is still in use. Launching project Postgres on ${alternatePort} instead." -ForegroundColor Yellow
            $Port = [int]$alternatePort
            $script:postgresPort = [int]$alternatePort
        }
    }

    if (-not (Ensure-PostgresRuntime)) {
        Write-Host "Failed to provision Postgres runtime automatically." -ForegroundColor Red
        Write-Host "Install Docker Desktop or PostgreSQL tools (initdb + pg_ctl), then rerun." -ForegroundColor Yellow
        exit 1
    }

    if (-not (Test-DockerRuntimeAvailable)) {
        Write-Host "Docker runtime unavailable after bootstrap. Re-invoking setup postgres bootstrap..." -ForegroundColor Yellow
        try {
            & .\scripts\infra\setup.ps1 -PostgresOnly
        } catch {
        }
        if (-not (Wait-ForDockerRuntime -TimeoutSeconds 120)) {
            Write-Host "Failed to start Docker runtime automatically." -ForegroundColor Red
            Write-Host "Homerun requires Docker Desktop for launcher-managed Postgres on Windows." -ForegroundColor Yellow
            Write-Host "Enable virtualization (Hyper-V/WSL2) and rerun Homerun.bat." -ForegroundColor Yellow
            exit 1
        }
    }

    Write-Host "Starting Postgres..." -ForegroundColor Cyan

    # Ensure a firewall allow rule exists for postgres.exe BEFORE starting.
    # On Windows 11, the firewall can silently block the binary on loopback
    # if the UAC allow dialog was never shown or was dismissed.
    $pgBinDir = Find-PostgresBinDir
    if ($pgBinDir) {
        Ensure-PostgresFirewallRule -PostgresBinDir $pgBinDir
    }

    $dockerStarted = $false
    if (Wait-ForDockerRuntime -TimeoutSeconds 120) {
        $dockerStarted = Start-PostgresDocker -PgHost $PgHost -Port $Port -Db $Db -User $User -Password $Password -ContainerName $ContainerName -Image $Image -DataDir $DataDir
    }
    $resolvedDockerPort = if ($script:lastPostgresDockerPublishedPort) { [int]$script:lastPostgresDockerPublishedPort } else { [int]$Port }
    if ($dockerStarted -and (Wait-ForService -TargetHost $PgHost -Port $resolvedDockerPort -TimeoutSeconds 90)) {
        $script:postgresPort = [int]$resolvedDockerPort
        $script:postgresStartedByScript = $true
        $script:postgresStartMode = if ($script:lastPostgresContainerEngine -eq "docker-compose") { "docker-compose" } else { "docker" }
        $postgresRuntimeLabel = if ($script:postgresStartMode -eq "docker-compose") { "Docker Compose" } else { "Docker" }
        if ($resolvedDockerPort -ne $Port) {
            Write-Host "Postgres started via $postgresRuntimeLabel on ${PgHost}:${resolvedDockerPort} (requested ${Port})" -ForegroundColor Green
        } else {
            Write-Host "Postgres started via $postgresRuntimeLabel on ${PgHost}:${resolvedDockerPort}" -ForegroundColor Green
        }
        return
    }

    if ($script:lastPostgresDockerError) {
        Write-Host "Docker Postgres startup error: $script:lastPostgresDockerError" -ForegroundColor Yellow
    }
    Show-PostgresDockerDiagnostics -ContainerName $ContainerName

    Write-Host "Failed to start Postgres automatically." -ForegroundColor Red
    Write-Host "Launcher could not confirm Postgres is reachable on ${PgHost}:${resolvedDockerPort}." -ForegroundColor Yellow
    Write-Host "Resolve the connectivity issue shown above, then rerun Homerun.bat." -ForegroundColor Yellow
    exit 1
}

function Cleanup-StartedPostgres {
    if (-not $script:postgresStartedByScript) {
        return
    }

    if ($script:postgresStartMode -eq "docker-compose") {
        $composeStop = Invoke-DockerComposeCommand -Arguments @("stop", "postgres") -EnvironmentOverrides @{
            "POSTGRES_CONTAINER_NAME" = $postgresContainerName
        }
        if ($composeStop.ExitCode -ne 0) {
            if (Ensure-DockerCommand) {
                try { docker stop $postgresContainerName *> $null } catch {}
            }
        }
        return
    }

    if ($script:postgresStartMode -eq "docker") {
        if (Ensure-DockerCommand) {
            try { docker stop $postgresContainerName *> $null } catch {}
            if ($script:postgresDockerCreatedByScript) {
                try { docker rm $postgresContainerName *> $null } catch {}
            }
        }
        return
    }

    if ($script:postgresStartMode -eq "local") {
        $binDir = Find-PostgresBinDir
        if ($binDir) {
            $pgctlPath = Join-Path $binDir "pg_ctl.exe"
            if (Test-Path $pgctlPath) {
                try { & $pgctlPath -D $postgresDataDir -m fast -w stop *> $null } catch {}
            }
        }
    }
}

function Test-DockerContainerRunningByName {
    param([string]$ContainerName)

    if (-not (Ensure-DockerCommand)) {
        return $false
    }

    $runningResult = Invoke-DockerCommand -Arguments @("inspect", "-f", "{{.State.Running}}", $ContainerName)
    if ($runningResult.ExitCode -ne 0) {
        return $false
    }

    return (((($runningResult.Output | Out-String).Trim()).ToLowerInvariant()) -eq "true")
}

function Cleanup-AdoptedDockerPostgres {
    if ($script:postgresStartedByScript) {
        return
    }

    if (-not (Test-DockerContainerRunningByName -ContainerName $postgresContainerName)) {
        return
    }

    $composeStop = Invoke-DockerComposeCommand -Arguments @("stop", "postgres") -EnvironmentOverrides @{
        "POSTGRES_CONTAINER_NAME" = $postgresContainerName
    }
    if ($composeStop.ExitCode -eq 0) {
        return
    }

    if (Ensure-DockerCommand) {
        try { docker stop $postgresContainerName *> $null } catch {}
    }
}

function Cleanup-StaleHomerunProcesses {
    <#
    .SYNOPSIS
    Kill orphaned Python worker processes from a previous crashed run.

    Finds python.exe processes whose command line contains "workers.runner"
    or "uvicorn" running from this project's backend directory and kills them.
    This prevents stale connections from saturating local services and blocking
    startup after an unclean exit.
    #>
    $projectRoot = (Get-Location).Path
    $backendDir = Join-Path $projectRoot "backend"

    try {
        $pythonProcesses = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue
    } catch {
        return
    }

    if (-not $pythonProcesses) {
        return
    }

    $killed = 0
    foreach ($proc in $pythonProcesses) {
        $cmdLine = $proc.CommandLine
        if (-not $cmdLine) { continue }

        # Only kill homerun-related processes (workers, uvicorn backend, tui)
        $isHomerun = $false
        if ($cmdLine -match "workers\.host") { $isHomerun = $true }
        elseif ($cmdLine -match "workers\.runner") { $isHomerun = $true }
        elseif ($cmdLine -match "workers\.\w+_worker") { $isHomerun = $true }
        elseif (($cmdLine -match "uvicorn") -and ($cmdLine -match "main:app")) { $isHomerun = $true }
        elseif ($cmdLine -match "tui\.py") { $isHomerun = $true }

        if (-not $isHomerun) { continue }

        # Verify the process is running from this project (not another instance)
        if ($cmdLine -notmatch [regex]::Escape($backendDir) -and
            $cmdLine -notmatch [regex]::Escape($projectRoot)) {
            continue
        }

        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
            $killed++
        } catch {}
    }

    if ($killed -gt 0) {
        Write-Host "Cleaned up $killed stale Homerun process(es) from a previous run." -ForegroundColor Yellow
        Start-Sleep -Seconds 1
    }
}

function Cleanup-LocalPostgresIfOwned {
    <#
    .SYNOPSIS
    Stop the launcher-managed local Postgres if it's running from our data directory.
    Called during shutdown even if the launcher didn't start it this session
    (i.e. it was already running when we launched).
    #>
    $binDir = Find-PostgresBinDir
    if (-not $binDir) { return }
    $pgctlPath = Join-Path $binDir "pg_ctl.exe"
    if (-not (Test-Path $pgctlPath)) { return }

    if (Test-LocalPostgresListenerOwned -DataDir $postgresDataDir -Port $script:postgresPort) {
        try { & $pgctlPath -D $postgresDataDir -m fast -w stop *> $null } catch {}
    }
}

function Invoke-GitCommand {
    param([string[]]$Arguments)

    $result = [ordered]@{
        ExitCode = 1
        Output = ""
    }

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        $result.Output = "Git CLI unavailable in current shell."
        return [pscustomobject]$result
    }

    $nativePrefExists = $false
    $previousNativePref = $null
    try {
        if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
            $nativePrefExists = $true
            $previousNativePref = $PSNativeCommandUseErrorActionPreference
            $PSNativeCommandUseErrorActionPreference = $false
        }
    } catch {
    }

    try {
        $output = (& git @Arguments 2>&1 | Out-String).Trim()
        $result.ExitCode = $LASTEXITCODE
        $result.Output = $output
    } catch {
        $result.ExitCode = if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { $LASTEXITCODE } else { 1 }
        $result.Output = $_.Exception.Message
    } finally {
        if ($nativePrefExists) {
            $PSNativeCommandUseErrorActionPreference = $previousNativePref
        }
    }

    return [pscustomobject]$result
}

function Invoke-AutoUpdateRepository {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "Git is not installed; skipping repository auto-update." -ForegroundColor Yellow
        return
    }

    $repoCheck = Invoke-GitCommand -Arguments @("rev-parse", "--is-inside-work-tree")
    $repoCheckValue = ($repoCheck.Output | Out-String).Trim().ToLowerInvariant()
    if ($repoCheck.ExitCode -ne 0 -or $repoCheckValue -ne "true") {
        return
    }

    $branchResult = Invoke-GitCommand -Arguments @("branch", "--show-current")
    $branch = ($branchResult.Output | Out-String).Trim()
    if ($branchResult.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($branch)) {
        Write-Host "Detached HEAD detected; skipping repository auto-update." -ForegroundColor Yellow
        return
    }

    $dirtyResult = Invoke-GitCommand -Arguments @("status", "--porcelain", "--untracked-files=no")
    if ($dirtyResult.ExitCode -ne 0) {
        Write-Host "Could not read git working tree status; skipping repository auto-update." -ForegroundColor Yellow
        return
    }
    if (-not [string]::IsNullOrWhiteSpace($dirtyResult.Output)) {
        Write-Host "Local tracked changes detected; skipping repository auto-update." -ForegroundColor Yellow
        return
    }

    $remoteName = "origin"
    $remoteBranch = $branch
    $upstreamResult = Invoke-GitCommand -Arguments @("rev-parse", "--abbrev-ref", "--symbolic-full-name", '@{u}')
    $upstreamRef = ($upstreamResult.Output | Out-String).Trim()
    if ($upstreamResult.ExitCode -eq 0 -and $upstreamRef.Contains("/")) {
        $splitIndex = $upstreamRef.IndexOf("/")
        if ($splitIndex -gt 0 -and $splitIndex -lt ($upstreamRef.Length - 1)) {
            $remoteName = $upstreamRef.Substring(0, $splitIndex)
            $remoteBranch = $upstreamRef.Substring($splitIndex + 1)
        }
    } else {
        $remoteRefCheck = Invoke-GitCommand -Arguments @("show-ref", "--verify", "--quiet", "refs/remotes/origin/$branch")
        if ($remoteRefCheck.ExitCode -ne 0) {
            Write-Host "No upstream branch configured for '$branch'; skipping repository auto-update." -ForegroundColor Yellow
            return
        }
    }

    Write-Host "Checking for code updates from ${remoteName}/${remoteBranch}..." -ForegroundColor Cyan
    $fetchResult = Invoke-GitCommand -Arguments @("-c", "credential.interactive=never", "fetch", "--quiet", $remoteName, $remoteBranch)
    if ($fetchResult.ExitCode -ne 0) {
        Write-Host "Unable to fetch updates; continuing with local copy." -ForegroundColor Yellow
        return
    }

    $pullResult = Invoke-GitCommand -Arguments @("-c", "credential.interactive=never", "pull", "--ff-only", "--no-rebase", $remoteName, $remoteBranch)
    if ($pullResult.ExitCode -eq 0) {
        if (($pullResult.Output | Out-String) -match "Already up[\s-]to date\.") {
            Write-Host "Code is up to date." -ForegroundColor Green
        } else {
            Write-Host "Code updated from ${remoteName}/${remoteBranch}." -ForegroundColor Green
        }
        return
    }

    Write-Host "Auto-update skipped (non fast-forward or local commits). Continuing with local copy." -ForegroundColor Yellow
}

function Test-NeedsSetup {
    if (-not (Test-Path "backend\venv")) { return $true }
    if (-not (Test-Path "backend\venv\Scripts\python.exe")) { return $true }
    if (-not (Test-Path "frontend\node_modules")) { return $true }
    if (-not (Test-Path ".setup-stamp.json")) { return $true }

    $venvPython = "backend\venv\Scripts\python.exe"
    try {
        & $venvPython -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12 else 1)" *> $null
        if ($LASTEXITCODE -ne 0) { return $true }
    } catch {
        return $true
    }
    try {
        & $venvPython -c "import py_clob_client, eth_account" *> $null
        if ($LASTEXITCODE -ne 0) { return $true }
    } catch {
        return $true
    }

    try {
        $stamp = Get-Content ".setup-stamp.json" -Raw | ConvertFrom-Json
    } catch {
        return $true
    }

    function Get-HashOrMissing {
        param([string]$Path)
        if (-not (Test-Path $Path)) { return "missing" }
        return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }

    $fingerprintPython = "backend\venv\Scripts\python.exe"

    $pythonVersion = (& $fingerprintPython -c "import platform; print(platform.python_version())")
    if ($stamp.python_version -ne $pythonVersion) { return $true }
    if ($stamp.requirements_sha256 -ne (Get-HashOrMissing "backend\requirements.txt")) { return $true }
    if ($stamp.requirements_trading_sha256 -ne (Get-HashOrMissing "backend\requirements-trading.txt")) { return $true }
    if ($stamp.package_json_sha256 -ne (Get-HashOrMissing "frontend\package.json")) { return $true }
    if ($stamp.package_lock_sha256 -ne (Get-HashOrMissing "frontend\package-lock.json")) { return $true }

    return $false
}

Invoke-AutoUpdateRepository

if (Test-NeedsSetup) {
    Write-Host "Setup missing or stale. Running setup..." -ForegroundColor Yellow
    & .\scripts\infra\setup.ps1
    if ($LASTEXITCODE -ne 0) {
        throw "Setup failed"
    }
    if (Test-NeedsSetup) {
        throw "Setup completed but required runtime artifacts are still missing or stale"
    }
}

# Kill orphaned workers from a previous crashed run before starting services.
# Stale processes hold local service connections that can block startup.
Cleanup-StaleHomerunProcesses

# If the previous run was terminated abruptly (window close/taskkill),
# launcher-named Docker services can remain running.
if (-not $script:databaseUrlWasProvided) {
    Cleanup-AdoptedDockerPostgres
}

# The Npcap Loopback Adapter (Wireshark/Nmap) can silently break loopback
# TCP connections, causing Postgres to appear to listen but drop all traffic.
Test-NpcapLoopbackInterference

try {
    if ($env:DATABASE_URL) {
        Write-Host "Using provided DATABASE_URL; skipping launcher-managed Postgres startup." -ForegroundColor Cyan
    } else {
        Ensure-Postgres -PgHost $postgresHost -Port $postgresPort -Db $postgresDb -User $postgresUser -Password $postgresPassword -ContainerName $postgresContainerName -Image $postgresImage -DataDir $postgresDataDir
        $env:DATABASE_URL = "postgresql+asyncpg://${postgresUser}:${postgresPassword}@${postgresHost}:${script:postgresPort}/${postgresDb}"
    }

    New-Item -ItemType Directory -Path "backend\.runtime" -Force | Out-Null
    Set-Content -Path "backend\.runtime\database_url" -Value $env:DATABASE_URL -Encoding UTF8

    & backend\venv\Scripts\python.exe .\scripts\infra\ensure_postgres_ready.py --database-url $env:DATABASE_URL --retries 240 --retry-delay-seconds 0.5
    if ($LASTEXITCODE -ne 0) {
        throw "Postgres readiness validation failed"
    }

    # Activate venv
    & backend\venv\Scripts\Activate.ps1

    # Ensure TUI dependencies are installed (pinned to Textual 8.x line)
    python -c "import textual; v=tuple(int(x) for x in textual.__version__.split('.')[:2]); exit(0 if v >= (8,0) and v < (9,0) else 1)" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing TUI dependencies..." -ForegroundColor Cyan
        python -m pip install -q "textual>=8.0.0,<9.0" "rich>=14.0.0,<15.0.0"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install TUI dependencies"
        }
    }

    if ($runServiceSmokeTest) {
        python .\scripts\infra\launcher_smoke.py
        exit $LASTEXITCODE
    }

    # Launch the TUI
    Ensure-ConsoleViewport
    try {
        Clear-Host
    } catch {
    }
    Start-Sleep -Milliseconds 250
    python tui.py @tuiArgs
} finally {
    # Kill any remaining Homerun Python processes (workers, backend, etc.)
    Cleanup-StaleHomerunProcesses

    # Stop launcher-managed services
    Cleanup-StartedPostgres

    # Stop adopted launcher-named Docker services as well. This handles
    # runs where services were already up from a previous session.
    if (-not $script:databaseUrlWasProvided) {
        Cleanup-AdoptedDockerPostgres
    }

    # Also stop services the launcher adopted (already running when we started)
    if (-not $script:databaseUrlWasProvided) {
        Cleanup-LocalPostgresIfOwned
    }
}

